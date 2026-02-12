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
from .repository import create_compositions, create_users
from .repository import create_table_compositions


def _fingerprint_value(value: Any) -> str:
    """
    Create a stable, comparable fingerprint for values used to link outputs->inputs.
    Supports scalars, dicts, and lists (via JSON canonicalization).
    """
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _extract_dataset_ids_from_value(value: Any) -> List[Any]:
    """
    Recursively extract dataset_id values from nested structures.

    Examples encountered in real logs:
    - {"theme": {"dataset_id": 3096, ...}}
    - {"new_table": {"dataset_id": "3086", ...}, "tables": [{"dataset_id": "..."}]}
    """
    out: List[Any] = []

    # If JSON-string, attempt to parse
    if isinstance(value, str):
        parsed = safe_json_parse(value, None)
        if parsed is not None and parsed != value:
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


def _build_table_compositions_from_calls(
    serializable_compositions: List[Dict[str, Any]],
    dataset_links: Dict[int, List[str]],
    dataset_outputs: Dict[int, List[str]],
    call_edges: Dict,
    calls_list: list,
    call_id_to_index: Dict[int, int],
) -> List[Dict[str, Any]]:
    """
    Build TableCompositions directly from recovery data structures.
    No intermediate extraction step — uses dataset_links, call_edges directly from Calls.

    Each TableComposition stores:
    - table_ids: which tables participate
    - service_mids: which services are in the chain
    - call_ids: traceability
    - join_steps: where a service consumes both a table and upstream output
    - nodes, links: full DAG for audit
    """
    from datetime import datetime

    table_compositions: List[Dict[str, Any]] = []

    for comp in serializable_compositions:
        comp_id = comp.get("id")
        nodes = comp.get("nodes") or []
        links = comp.get("links") or []

        # Separate table nodes vs call nodes
        table_ids: List[int] = []
        call_ids: List[int] = []
        service_mids: List[int] = []
        call_id_to_mid: Dict[int, int] = {}
        owner = None
        times: List[datetime] = []

        for node in nodes:
            mid = node.get("mid")
            nid = node.get("id")

            if mid is None:
                # Table/dataset node
                try:
                    table_ids.append(int(str(nid)))
                except Exception:
                    pass
            else:
                try:
                    cid = int(str(nid))
                    mid_int = int(str(mid))
                    call_ids.append(cid)
                    service_mids.append(mid_int)
                    call_id_to_mid[cid] = mid_int
                    if not owner and node.get("owner"):
                        owner = node["owner"]
                    st = node.get("start_time")
                    if st:
                        try:
                            times.append(datetime.fromisoformat(st))
                        except Exception:
                            pass
                except Exception:
                    pass

        # Deduplicate preserving order
        seen = set()
        table_ids = [x for x in table_ids if not (x in seen or seen.add(x))]
        seen.clear()
        call_ids = [x for x in call_ids if not (x in seen or seen.add(x))]
        seen.clear()
        service_mids = [x for x in service_mids if not (x in seen or seen.add(x))]

        start_time = min(times) if times else None
        end_time = max(times) if times else None
        table_ids_set = set(table_ids)

        # Build join_steps directly from links
        links_by_target: Dict[str, List[Dict]] = {}
        for link in links:
            tgt = str(link.get("target", ""))
            links_by_target.setdefault(tgt, []).append(link)

        join_steps: List[Dict[str, Any]] = []
        for cid in call_ids:
            incoming = links_by_target.get(str(cid), [])
            table_inputs = []
            upstream_calls = []

            for link in incoming:
                src = link.get("source")
                if src is None:
                    continue
                try:
                    src_int = int(str(src))
                except Exception:
                    continue

                if src_int in table_ids_set:
                    table_inputs.append({"table_id": src_int, "fields": link.get("fields")})
                elif src_int in call_id_to_mid:
                    upstream_calls.append({
                        "source_call_id": src_int,
                        "source_service_mid": call_id_to_mid.get(src_int),
                        "fields": link.get("fields"),
                    })

            if not table_inputs:
                continue

            join_steps.append({
                "target_call_id": cid,
                "target_service_mid": call_id_to_mid.get(cid),
                "table_inputs": table_inputs,
                "upstream_calls": upstream_calls,
                "is_join": bool(upstream_calls),
            })

        if not comp_id:
            continue

        table_compositions.append({
            "id": comp_id,
            "owner": owner,
            "start_time": start_time,
            "end_time": end_time,
            "table_ids": table_ids,
            "call_ids": call_ids,
            "service_mids": service_mids,
            "join_steps": join_steps,
            "nodes": nodes,
            "links": links,
        })

    return table_compositions


def _is_successful_with_wms(task: Call, result_data: Dict) -> bool:
    """
    Check if task is successful WMS task or mapcombine task
    
    Args:
        task: Task/Call object
        result_data: Parsed result data
    
    Returns:
        True if task is successful with WMS or mapcombine
    """
    return (
        result_data and
        ((result_data.get("status") == "success" and "wms_link" in result_data) or 
         (task.mid == 399 and "map" in result_data and task.status == TASK_SUCCEEDED))
    )


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
        
        # For 'edit' widget type, convert value to string
        if widget_type == WIDGET_EDIT and input_value is not None:
            input_value = str(input_value)
        
        # Skip if input_value is not hashable (dict, list, etc.)
        if input_value and is_hashable(input_value):
            if input_value in file_value_tracker and widget_type != WIDGET_THEME_SELECT:
                tracker_info = file_value_tracker[input_value]
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
        
        # For 'edit' widget type, convert value to string and track
        if widget_type == WIDGET_EDIT and output_value is not None:
            output_value = str(output_value)
            file_value_tracker[output_value] = {
                "value": task.id,
                "name": param_name
            }
        # Track hashable values for other widget types
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
        
        # Build users dict
        for task in tasks_list:
            if task.owner:
                users[task.owner] = True
        
        print(f"Processing {len(tasks_list)} tasks...")
        
        # Main processing loop
        for task in tasks_list:
            inputs = safe_json_parse(task.input, {})
            result_data = safe_json_parse(task.result, {})
            
            if task.mid not in in_and_out:
                continue
            
            service_inputs = in_and_out[task.mid].get("input", {})
            service_outputs = in_and_out[task.mid].get("output", {})
            
            # Check if successful with WMS
            is_successful_with_wms = _is_successful_with_wms(task, result_data)
            
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
        
        return {
            "success": True,
            "message": "Service composition recovery completed",
            "compositionsCount": len(compositions),
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
        # call_id -> list of (normalized_dataset_id, param_name) for INPUTS
        dataset_links: Dict[int, List[str]] = {}
        # call_id -> list of (normalized_dataset_id, param_name) for OUTPUTS
        dataset_outputs: Dict[int, List[str]] = {}
        # dataset_id -> latest producing call_id (only if produced before consumer)
        dataset_producers: Dict[str, int] = {}
        service_dataset_edges = {}
        # fingerprint -> {source_call_id, source_param_name}
        file_tracker: Dict[str, Dict[str, Any]] = {}
        call_edges = {}
        call_id_to_index = {call.id: idx for idx, call in enumerate(calls_list)}
        users = {call.owner: True for call in calls_list if call.owner}
        
        print(f"Processing {len(calls_list)} calls...")
        
        # First pass: analyze connections
        for call in calls_list:
            if call.status != TASK_SUCCEEDED:
                continue
            
            inputs = safe_json_parse(call.input, {})
            outputs = safe_json_parse(call.result, {})
            
            service_inputs = in_and_out.get(call.mid, {}).get("input", {}) if in_and_out else {}
            service_outputs = in_and_out.get(call.mid, {}).get("output", {}) if in_and_out else {}
            
            # Process inputs
            # 1) Always scan ALL inputs for dataset_id (table references)
            # 2) Use configured input params (if available) to improve fingerprint linking
            input_items = []
            if isinstance(inputs, dict):
                input_items = list(inputs.items())

            configured_keys = list(service_inputs.keys()) if service_inputs else []
            keys_to_scan = configured_keys if configured_keys else [k for k, _ in input_items]

            # Dataset ids: scan all input values
            for param_name, input_value in input_items:
                if input_value is None:
                    continue
                dataset_ids = _extract_dataset_ids_from_value(input_value)
                for raw_dataset_id in dataset_ids:
                    try:
                        normalized_id = normalize_dataset_id(raw_dataset_id, guid_map)
                    except Exception:
                        continue

                    dataset_links.setdefault(call.id, []).append(f"{normalized_id}:{param_name}")

                    # Update service-dataset edges stats
                    if normalized_id not in service_dataset_edges:
                        service_dataset_edges[normalized_id] = {}
                    if call.mid not in service_dataset_edges[normalized_id]:
                        service_dataset_edges[normalized_id][call.mid] = {"total": 0}
                    if call.owner not in service_dataset_edges[normalized_id][call.mid]:
                        service_dataset_edges[normalized_id][call.mid][call.owner] = 0

                    service_dataset_edges[normalized_id][call.mid][call.owner] += 1
                    service_dataset_edges[normalized_id][call.mid]["total"] += 1

            # Fingerprint linking: scan ALL input keys (configured keys may not match real log fields)
            all_input_keys = list(inputs.keys()) if isinstance(inputs, dict) else []
            for param_name in all_input_keys:
                input_value = inputs.get(param_name)
                if input_value is None:
                    continue

                widget_type = service_inputs.get(param_name)

                # File/edit connection: use configured widget types if known; otherwise fingerprint everything
                should_check_fingerprint = (
                    widget_type in (WIDGET_FILE, WIDGET_EDIT) or widget_type is None
                )
                if should_check_fingerprint:
                    fp = _fingerprint_value(input_value)
                    if not fp:
                        continue
                    file_info = file_tracker.get(fp)
                    if file_info and file_info.get("source_call_id") and file_info.get("source_param_name"):
                        if call.id not in call_edges:
                            call_edges[call.id] = {}
                        if file_info["source_call_id"] not in call_edges[call.id]:
                            call_edges[call.id][file_info["source_call_id"]] = []
                        call_edges[call.id][file_info["source_call_id"]].append(
                            f"{file_info['source_param_name']}:{param_name}"
                        )
            
            # Register output values for tracking — scan ALL output keys (configured may not match real log fields)
            all_output_keys = list(outputs.keys()) if isinstance(outputs, dict) else []

            for param_name in all_output_keys:
                output_value = outputs.get(param_name) if isinstance(outputs, dict) else None
                if output_value is None:
                    continue

                # Dataset outputs (derived tables)
                output_dataset_ids = _extract_dataset_ids_from_value(output_value)
                for raw_dataset_id in output_dataset_ids:
                    try:
                        normalized_id = normalize_dataset_id(raw_dataset_id, guid_map)
                    except Exception:
                        continue
                    dataset_outputs.setdefault(call.id, []).append(f"{normalized_id}:{param_name}")
                    dataset_producers[str(normalized_id)] = call.id

                fp = _fingerprint_value(output_value)
                if not fp:
                    continue
                file_tracker[fp] = {
                    "source_call_id": call.id,
                    "source_param_name": param_name
                }
        
        # Second pass: build compositions
        raw_compositions = {}
        
        for call in calls_list:
            if call.status != TASK_SUCCEEDED:
                continue
            
            # Dataset connections (can be multiple per call)
            if call.id in dataset_links:
                nodes = []
                links = []
                for entry in dataset_links[call.id]:
                    try:
                        dataset_id_str, param_name = entry.split(':', 1)
                    except Exception:
                        continue
                    dataset_node = {
                        "id": dataset_id_str,
                        "start_date": call.start_time.isoformat() if call.start_time else None
                    }
                    dataset_link = {
                        "source": dataset_id_str,
                        "target": call.id,
                        "fields": f"{dataset_id_str}:{param_name}"
                    }
                    nodes.append(dataset_node)
                    links.append(dataset_link)

                    # If dataset was produced by a prior call, link producer -> dataset
                    producer_call_id = dataset_producers.get(dataset_id_str)
                    if producer_call_id and producer_call_id < call.id:
                        links.append(
                            {
                                "source": producer_call_id,
                                "target": dataset_id_str,
                                "fields": f"{dataset_id_str}:produced"
                            }
                        )
                        # Ensure producer node is present
                        producer_call = calls_list[call_id_to_index[producer_call_id]]
                        nodes.append(producer_call)

                raw_compositions[call.id] = {
                    "nodes": nodes + [call],
                    "links": links
                }

            # Dataset outputs (call -> dataset)
            if call.id in dataset_outputs:
                nodes = []
                links = []
                for entry in dataset_outputs[call.id]:
                    try:
                        dataset_id_str, param_name = entry.split(':', 1)
                    except Exception:
                        continue
                    dataset_node = {
                        "id": dataset_id_str,
                        "start_date": call.start_time.isoformat() if call.start_time else None
                    }
                    dataset_link = {
                        "source": call.id,
                        "target": dataset_id_str,
                        "fields": f"{param_name}:dataset_id"
                    }
                    nodes.append(dataset_node)
                    links.append(dataset_link)

                if call.id in raw_compositions:
                    raw_compositions[call.id]["nodes"].extend(nodes)
                    raw_compositions[call.id]["links"].extend(links)
                else:
                    raw_compositions[call.id] = {
                        "nodes": nodes + [call],
                        "links": links
                    }
            
            # Call edges
            if call.id in call_edges:
                # IMPORTANT:
                # A call may have multiple upstream inputs (multiple source_call_id),
                # e.g. mapcombine-like operations. We must NOT overwrite the composition per source.
                # Instead, we merge (union) nodes/links from all upstream branches.

                merged_nodes = []
                merged_links = []

                # Start from dataset link composition if it exists
                if call.id in raw_compositions:
                    merged_nodes.extend(raw_compositions[call.id]["nodes"])
                    merged_links.extend(raw_compositions[call.id]["links"])

                # Merge all upstream branches
                for source_call_id, fields in call_edges[call.id].items():
                    link = {"source": source_call_id, "target": call.id, "fields": fields}

                    if source_call_id in raw_compositions:
                        merged_nodes.extend(raw_compositions[source_call_id]["nodes"])
                        merged_links.extend(raw_compositions[source_call_id]["links"])
                    else:
                        source_call = calls_list[call_id_to_index[source_call_id]]
                        merged_nodes.append(source_call)

                    merged_links.append(link)

                merged_nodes.append(call)

                # Deduplicate nodes while preserving order
                seen = set()
                unique_nodes = []
                for n in merged_nodes:
                    if hasattr(n, "mid"):
                        key = ("call", int(n.id))
                    elif isinstance(n, dict) and "id" in n:
                        key = ("dataset", str(n["id"]))
                    else:
                        key = ("other", str(n))

                    if key in seen:
                        continue
                    seen.add(key)
                    unique_nodes.append(n)

                # Deduplicate links (stable JSON-ish signature)
                seen_links = set()
                unique_links = []
                for l in merged_links:
                    sig = (
                        str(l.get("source")),
                        str(l.get("target")),
                        json.dumps(l.get("fields"), sort_keys=True, ensure_ascii=False)
                        if isinstance(l.get("fields"), (dict, list))
                        else str(l.get("fields")),
                    )
                    if sig in seen_links:
                        continue
                    seen_links.add(sig)
                    unique_links.append(l)

                raw_compositions[call.id] = {"nodes": unique_nodes, "links": unique_links}
        
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
        
        # Convert compositions to JSON-serializable format AND assign stable IDs
        serializable_compositions: List[Dict[str, Any]] = []
        for i, comp in enumerate(final_compositions):
            try:
                print(f"Processing composition {i+1}/{len(final_compositions)}, nodes count: {len(comp['nodes'])}")

                serializable_nodes: List[Dict[str, Any]] = []
                call_id_chain: List[int] = []

                for j, node in enumerate(comp["nodes"]):
                    try:
                        if hasattr(node, "mid"):
                            # Call object
                            call_id_chain.append(int(node.id))
                            node_dict = {
                                "id": int(node.id),
                                "mid": int(node.mid) if node.mid is not None else None,
                                "owner": node.owner,
                                "start_time": node.start_time.isoformat() if node.start_time else None,
                            }
                        else:
                            # Dataset node dict
                            node_id = node.get("id") if isinstance(node, dict) else str(node)
                            node_dict = {
                                "id": str(node_id),
                                "mid": None,
                                "owner": None,
                                "start_time": node.get("start_date") if isinstance(node, dict) else None,
                            }
                        serializable_nodes.append(node_dict)
                    except Exception as e:
                        print(f"Error processing node {j} in composition {i}: {e}, node type: {type(node)}")
                        raise

                comp_id = "_".join(map(str, call_id_chain)) if call_id_chain else None
                if not comp_id:
                    # Extremely defensive: skip if we cannot build a stable id
                    continue

                serializable_comp = {
                    "id": comp_id,
                    "nodes": serializable_nodes,
                    "links": comp["links"],
                }
                serializable_compositions.append(serializable_comp)
            except Exception as e:
                print(f"Error processing composition {i}: {e}")
                raise

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_compositions, f, indent=2)
        print(f"Successfully saved compositions to {output_path}")

        # Persist recovered compositions into DB for API access (/compositions/)
        await create_compositions(db, serializable_compositions)

        # Build TableCompositions directly from recovery data (no intermediate extraction)
        table_compositions = _build_table_compositions_from_calls(
            serializable_compositions, dataset_links, dataset_outputs, call_edges, calls_list, call_id_to_index
        )
        await create_table_compositions(db, table_compositions)
        
        print(f"Created {len(final_compositions)} final compositions")
        
        return {
            "success": True,
            "message": "Advanced composition recovery completed",
            "compositionsCount": len(final_compositions),
            "tableCompositionsCount": len(table_compositions),
            "serviceSequencesCount": len(longest_service_sequences),
            "servicesCount": len(in_and_out),
            "datasetsCount": len(service_dataset_edges)
        }
        
    except Exception as e:
        print(f"Error in recover_new function: {e}")
        raise

