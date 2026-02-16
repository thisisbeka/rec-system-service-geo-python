"""
Composition recovery algorithms
"""
import json
from typing import Dict, Any, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Call
from app.services.utils.constants import (
    WIDGET_THEME_SELECT, WIDGET_FILE, WIDGET_EDIT, TASK_SUCCEEDED
)
from app.services.utils.parsers import safe_json_parse
from app.services.utils.validators import is_hashable
from .service_map import build_service_connection_map, build_dataset_guid_map
from .builder import build_composition_for_task, normalize_composition
from .helpers import add_task_link, normalize_dataset_id
from .repository import create_compositions, create_users, create_table_compositions

SUCCESS_STATUSES = {"TASK_SUCCEEDED", "TASK_SUCCESS", "SUCCESS", "SUCCEEDED"}


def _normalize_link_value(value: Any) -> Any:
    """
    Normalize values used for linking between calls.
    Works for scalars, arrays, objects and JSON-encoded strings.
    """
    def _canonicalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _canonicalize(v) for k, v in sorted(obj.items(), key=lambda item: str(item[0]))}
        if isinstance(obj, list):
            # Preserve original sequence: list order is semantically important for linkage.
            return [_canonicalize(item) for item in obj]
        return obj

    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        # Avoid treating plain text values as JSON (prevents noisy parse errors).
        if stripped.startswith("{") or stripped.startswith("["):
            parsed = safe_json_parse(stripped, stripped)
        else:
            parsed = stripped
    else:
        parsed = value

    parsed = _canonicalize(parsed)

    if isinstance(parsed, (list, dict)):
        try:
            return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(parsed)

    return str(parsed)


def _extract_dataset_ids_from_value(value: Any) -> List[Any]:
    """Recursively extract dataset_id values from nested input structures."""
    out: List[Any] = []

    if isinstance(value, str):
        parsed = safe_json_parse(value, value)
        if parsed is not value:
            value = parsed

    if isinstance(value, dict):
        if "dataset_id" in value:
            out.append(value.get("dataset_id"))
        for v in value.values():
            out.extend(_extract_dataset_ids_from_value(v))
    elif isinstance(value, list):
        for item in value:
            out.extend(_extract_dataset_ids_from_value(item))

    return out


def _is_success_status(status: Any) -> bool:
    """Return True for known success statuses; empty status is treated as unknown (allowed)."""
    if status is None:
        return True
    normalized = str(status).strip().upper()
    if not normalized:
        return True
    return normalized in SUCCESS_STATUSES


def _is_successful_with_wms(task: Call, result_data: Dict, service_outputs: Dict[str, Any]) -> bool:
    """
    Check if task is successful WMS task or mapcombine task
    
    Args:
        task: Task/Call object
        result_data: Parsed result data
    
    Returns:
        True if task is successful with WMS or mapcombine
    """
    if not result_data:
        return False

    if not _is_success_status(task.status):
        return False

    # Legacy terminal marker for WMS services.
    if result_data.get("status") == "success" and "wms_link" in result_data:
        return True

    # Generic terminal marker driven by in_and_out configuration:
    # if any configured output field is present/non-empty, treat as terminal call.
    for output_name, widget_type in service_outputs.items():
        output_value = result_data.get(output_name)
        value_for_check = (
            _normalize_link_value(output_value)
            if widget_type == WIDGET_EDIT
            else output_value
        )
        if value_for_check not in (None, "", [], {}, "[]", "{}"):
            return True

    return False


def _process_task_inputs(task: Call, inputs: Dict, service_inputs: Dict,
                        file_value_tracker: Dict, task_links: Dict) -> None:
    """
    Process task inputs to find links to previous tasks
    
    Args:
        task: Current task
        inputs: Task input data
        service_inputs: Service input configuration
        file_value_tracker: Tracker of file values
        task_links: Dictionary to store found links
    """
    for param_name in service_inputs.keys():
        input_value = inputs.get(param_name) if isinstance(inputs, dict) else None
        widget_type = service_inputs[param_name]
        if widget_type == WIDGET_THEME_SELECT:
            continue

        input_key = (
            _normalize_link_value(input_value)
            if widget_type == WIDGET_EDIT
            else input_value
        )

        if input_key and is_hashable(input_key) and input_key in file_value_tracker:
            tracker_info = file_value_tracker[input_key]
            add_task_link(
                task_links,
                task.id,
                tracker_info["value"],
                tracker_info["name"],
                param_name
            )


def _register_task_outputs(task: Call, result_data: Dict, service_outputs: Dict,
                          file_value_tracker: Dict) -> None:
    """
    Register task outputs for tracking
    
    Args:
        task: Current task
        result_data: Task result data
        service_outputs: Service output configuration
        file_value_tracker: Tracker to store output values
    """
    if not (result_data and isinstance(result_data, dict)):
        return
    
    for param_name in service_outputs.keys():
        output_value = result_data.get(param_name)
        widget_type = service_outputs[param_name]

        if widget_type == WIDGET_EDIT:
            output_key = _normalize_link_value(output_value)
            if output_key is None:
                continue
            file_value_tracker[output_key] = {
                "value": task.id,
                "name": param_name
            }
        elif output_value and is_hashable(output_value):
            file_value_tracker[output_value] = {
                "value": task.id,
                "name": param_name
            }


async def recover(db: AsyncSession) -> Dict[str, Any]:
    """
    Recover service compositions from call history
    Original algorithm
    
    Args:
        db: Database session
    
    Returns:
        Dictionary with recovery results
    """
    try:
        print("Starting service composition recovery...")
        
        # Build service connection map
        in_and_out = await build_service_connection_map(db)
        guid_map = await build_dataset_guid_map(db)
        
        # Get all tasks
        print("Loading tasks...")
        result = await db.execute(select(Call).order_by(Call.id.asc()))
        tasks = result.scalars().all()
        tasks_list = list(tasks)
        
        compositions = []
        file_value_tracker = {}
        task_links = {}
        task_id_to_index = {task.id: idx for idx, task in enumerate(tasks_list)}
        users = {}
        table_compositions = []
        
        # Build users dict
        for task in tasks_list:
            if task.owner:
                users[task.owner] = True
        
        print(f"Processing {len(tasks_list)} tasks...")
        
        # Main processing loop
        for task in tasks_list:
            inputs = safe_json_parse(task.input, {})
            result_data = safe_json_parse(task.result, {})

            # Build minimal table compositions directly from call logs:
            # one row per successful call that references table dataset_id(s) in input.
            if _is_success_status(task.status):
                raw_dataset_ids = _extract_dataset_ids_from_value(inputs)
                normalized_tables: List[int] = []
                seen_tables = set()
                for raw_dataset_id in raw_dataset_ids:
                    try:
                        table_id = normalize_dataset_id(raw_dataset_id, guid_map)
                    except Exception:
                        continue
                    if table_id in seen_tables:
                        continue
                    seen_tables.add(table_id)
                    normalized_tables.append(table_id)
                if normalized_tables:
                    table_compositions.append({
                        "id": str(task.id),
                        "table_ids": normalized_tables,
                    })
            
            if task.mid not in in_and_out:
                continue
            
            service_inputs = in_and_out[task.mid].get("input", {})
            service_outputs = in_and_out[task.mid].get("output", {})
            
            # Check if successful with WMS
            is_successful_with_wms = _is_successful_with_wms(task, result_data, service_outputs)
            
            if is_successful_with_wms:
                # Process inputs to find links
                _process_task_inputs(task, inputs, service_inputs, file_value_tracker, task_links)
                
                # Build composition
                comp_data = build_composition_for_task(
                    task, task_links, tasks_list, task_id_to_index, in_and_out
                )
                
                nodes_count = len(comp_data["nodes"])
                if nodes_count > 1:
                    composition = normalize_composition(comp_data["nodes"], comp_data["localLinks"])
                    compositions.append(composition)
            else:
                # Process intermediate task inputs
                _process_task_inputs(task, inputs, service_inputs, file_value_tracker, task_links)
            
            # Register output files (for ALL tasks, not just intermediate ones)
            _register_task_outputs(task, result_data, service_outputs, file_value_tracker)
        
        print(f"Created {len(compositions)} compositions")
        
        # Save results
        await create_compositions(db, compositions)
        await create_users(db, users)
        await create_table_compositions(db, table_compositions)
        
        return {
            "success": True,
            "message": "Service composition recovery completed",
            "compositionsCount": len(compositions),
            "tableCompositionsCount": len(table_compositions),
            "usersCount": len(users)
        }
        
    except Exception as e:
        print(f"Error in recover function: {e}")
        raise


async def recover_new(db: AsyncSession) -> Dict[str, Any]:
    """
    Advanced service composition recovery
    Improved algorithm with dataset tracking
    
    Args:
        db: Database session
    
    Returns:
        Dictionary with recovery results
    """
    try:
        print("Starting advanced service composition recovery...")
        
        # Build maps in parallel
        in_and_out = await build_service_connection_map(db)
        guid_map = await build_dataset_guid_map(db)
        
        # Get all calls
        print("Loading calls...")
        result = await db.execute(select(Call).order_by(Call.id.asc()))
        calls = result.scalars().all()
        calls_list = list(calls)
        
        # Initialize data structures
        dataset_links = {}
        service_dataset_edges = {}
        file_tracker = {}
        call_edges = {}
        call_id_to_index = {call.id: idx for idx, call in enumerate(calls_list)}
        users = {call.owner: True for call in calls_list if call.owner}
        
        print(f"Processing {len(calls_list)} calls...")
        
        # First pass: analyze connections
        for call in calls_list:
            if not _is_success_status(call.status):
                continue
            
            inputs = safe_json_parse(call.input, {})
            outputs = safe_json_parse(call.result, {})
            
            if call.mid not in in_and_out:
                continue
            
            service_inputs = in_and_out[call.mid].get("input", {})
            service_outputs = in_and_out[call.mid].get("output", {})
            
            if not service_inputs or not service_outputs:
                continue
            
            # Process inputs
            for param_name in service_inputs.keys():
                input_value = inputs.get(param_name) if isinstance(inputs, dict) else None
                if not input_value:
                    continue
                
                widget_type = service_inputs[param_name]
                
                if widget_type == WIDGET_THEME_SELECT:
                    # Dataset connection
                    parsed_input = safe_json_parse(input_value, input_value) if isinstance(input_value, str) else input_value
                    
                    if isinstance(parsed_input, dict) and "dataset_id" in parsed_input:
                        normalized_id = normalize_dataset_id(parsed_input["dataset_id"], guid_map)
                        dataset_links[call.id] = f"{normalized_id}:{param_name}"
                        
                        # Update service-dataset edges
                        if normalized_id not in service_dataset_edges:
                            service_dataset_edges[normalized_id] = {}
                        if call.mid not in service_dataset_edges[normalized_id]:
                            service_dataset_edges[normalized_id][call.mid] = {"total": 0}
                        if call.owner not in service_dataset_edges[normalized_id][call.mid]:
                            service_dataset_edges[normalized_id][call.mid][call.owner] = 0
                        
                        service_dataset_edges[normalized_id][call.mid][call.owner] += 1
                        service_dataset_edges[normalized_id][call.mid]["total"] += 1
                        
                elif widget_type == WIDGET_FILE or widget_type == WIDGET_EDIT:
                    # File connection or edit widget
                    input_key = (
                        _normalize_link_value(input_value)
                        if widget_type == WIDGET_EDIT
                        else input_value
                    )
                    if not (input_key and is_hashable(input_key)):
                        continue
                    file_info = file_tracker.get(input_key)
                    if file_info and file_info.get("source_call_id") and file_info.get("source_param_name"):
                        if call.id not in call_edges:
                            call_edges[call.id] = {}
                        if file_info["source_call_id"] not in call_edges[call.id]:
                            call_edges[call.id][file_info["source_call_id"]] = []
                        
                        call_edges[call.id][file_info["source_call_id"]].append(
                            f"{file_info['source_param_name']}:{param_name}"
                        )
            
            # Register output files
            for param_name in service_outputs.keys():
                output_value = outputs.get(param_name) if isinstance(outputs, dict) else None
                widget_type = service_outputs[param_name]
                
                if widget_type == WIDGET_EDIT:
                    output_key = _normalize_link_value(output_value)
                    if output_key is None:
                        continue
                    file_tracker[output_key] = {
                        "source_call_id": call.id,
                        "source_param_name": param_name
                    }
                elif widget_type == WIDGET_FILE and output_value and is_hashable(output_value):
                    file_tracker[output_value] = {
                        "source_call_id": call.id,
                        "source_param_name": param_name
                    }
        
        # Second pass: build compositions
        raw_compositions = {}
        
        for call in calls_list:
            if not _is_success_status(call.status):
                continue
            
            # Dataset connections
            if call.id in dataset_links:
                dataset_id_str, param_name = dataset_links[call.id].split(':')
                
                dataset_node = {
                    "id": dataset_id_str,
                    "start_date": call.start_time.isoformat() if call.start_time else None
                }
                
                dataset_link = {
                    "source": dataset_id_str,
                    "target": call.id,
                    "fields": f"{dataset_id_str}:{param_name}"
                }
                
                raw_compositions[call.id] = {
                    "nodes": [dataset_node, call],
                    "links": [dataset_link]
                }
            
            # Call edges
            if call.id in call_edges:
                for source_call_id, fields in call_edges[call.id].items():
                    link = {"source": source_call_id, "target": call.id, "fields": fields}
                    
                    if source_call_id in raw_compositions:
                        # Inherit composition
                        raw_compositions[call.id] = {
                            "nodes": raw_compositions[source_call_id]["nodes"] + [call],
                            "links": raw_compositions[source_call_id]["links"] + [link]
                        }
                    else:
                        # Create new composition
                        source_call = calls_list[call_id_to_index[source_call_id]]
                        raw_compositions[call.id] = {
                            "nodes": [source_call, call],
                            "links": [link]
                        }
        
        # Extract sequences
        call_sequences = []
        service_sequences = []
        
        for composition in raw_compositions.values():
            call_ids = []
            service_ids = []
            
            for node in composition["nodes"]:
                if hasattr(node, 'mid'):  # It's a Call object
                    call_ids.append(node.id)
                    service_ids.append(node.mid)
            
            if call_ids:
                call_sequences.append('_'.join(map(str, call_ids)))
                service_sequences.append('_'.join(map(str, service_ids)))
        
        # Filter non-prefix sequences
        def filter_non_prefix(sequences):
            return [
                seq for i, seq in enumerate(sequences)
                if not any(i != j and other.startswith(seq) for j, other in enumerate(sequences))
            ]
        
        longest_call_sequences = filter_non_prefix(call_sequences)
        longest_service_sequences = filter_non_prefix(list(set(service_sequences)))
        
        # Build final compositions
        final_compositions = [
            raw_compositions[int(seq.split('_')[-1])]
            for seq in longest_call_sequences
            if int(seq.split('_')[-1]) in raw_compositions
        ]
        
        # Save compositions DAG to file
        output_path = "app/static/compositionsDAG.json"
        
        print(f"Preparing to save {len(final_compositions)} compositions to file...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Convert Call objects to dicts for JSON serialization
            serializable_compositions = []
            for i, comp in enumerate(final_compositions):
                try:
                    print(f"Processing composition {i+1}/{len(final_compositions)}, nodes count: {len(comp['nodes'])}")
                    
                    serializable_nodes = []
                    for j, node in enumerate(comp["nodes"]):
                        try:
                            node_dict = {
                                "id": node.id if hasattr(node, 'id') else (node.get("id") if isinstance(node, dict) else str(node)),
                                "mid": node.mid if hasattr(node, 'mid') else None,
                                "owner": node.owner if hasattr(node, 'owner') else None,
                                "start_time": node.start_time.isoformat() if hasattr(node, 'start_time') and node.start_time else (node.get("start_date") if isinstance(node, dict) else None)
                            }
                            serializable_nodes.append(node_dict)
                        except Exception as e:
                            print(f"Error processing node {j} in composition {i}: {e}, node type: {type(node)}")
                            raise
                    
                    serializable_comp = {
                        "nodes": serializable_nodes,
                        "links": comp["links"]
                    }
                    serializable_compositions.append(serializable_comp)
                except Exception as e:
                    print(f"Error processing composition {i}: {e}")
                    raise
            
            json.dump(serializable_compositions, f, indent=2)
            print(f"Successfully saved compositions to {output_path}")
        
        print(f"Created {len(final_compositions)} final compositions")
        
        return {
            "success": True,
            "message": "Advanced composition recovery completed",
            "compositionsCount": len(final_compositions),
            "serviceSequencesCount": len(longest_service_sequences),
            "servicesCount": len(in_and_out),
            "datasetsCount": len(service_dataset_edges)
        }
        
    except Exception as e:
        print(f"Error in recover_new function: {e}")
        raise
