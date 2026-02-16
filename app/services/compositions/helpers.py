"""
Helper functions for composition building
"""
from typing import Dict, Any, List
from app.services.utils.constants import DATASET_ID_OFFSET, WIDGET_THEME_SELECT
from app.services.utils.parsers import safe_json_parse


def normalize_dataset_id(dataset_id: Any, guid_map: Dict[str, int]) -> int:
    """
    Normalize dataset ID (convert GUID to ID and add offset)
    
    Args:
        dataset_id: Dataset ID (int) or GUID (string)
        guid_map: Mapping of GUIDs to IDs
    
    Returns:
        Normalized dataset ID with offset
    """
    normalized_id: Any = dataset_id
    
    # Convert GUID to numeric ID if possible
    if isinstance(dataset_id, str):
        if dataset_id in guid_map:
            normalized_id = guid_map[dataset_id]
        else:
            # Numeric string dataset ids are common in logs
            try:
                normalized_id = int(dataset_id)
            except Exception:
                # Unknown GUID/string: cannot normalize to int without a map
                # Callers should handle this (e.g. skip dataset link or store separately).
                raise ValueError(f"Unsupported dataset_id format (no guid_map entry): {dataset_id}")
    
    # Add offset to distinguish from service IDs
    return int(normalized_id) + DATASET_ID_OFFSET


def add_task_link(links: Dict, task_id: int, source_id: int, 
                  source_param_name: str, param_name: str) -> None:
    """
    Add a link between tasks
    
    Args:
        links: Dictionary of task links
        task_id: Target task ID
        source_id: Source task ID
        source_param_name: Source parameter name
        param_name: Target parameter name
    """
    task_id_str = str(task_id)
    link_data = {
        "source": source_id,
        "target": task_id,
        "value": f"{source_param_name}:{param_name}"
    }
    
    if task_id_str not in links:
        links[task_id_str] = [link_data]
    else:
        links[task_id_str].append(link_data)


def create_composition_node(task: Any, in_and_out: Dict[int, Any]) -> Dict[str, Any]:
    """
    Create composition node from task
    
    Args:
        task: Task/Call object
        in_and_out: Service connection map
    
    Returns:
        Composition node dictionary or None if service info not found
    """
    inputs = safe_json_parse(task.input, {})
    outputs = safe_json_parse(task.result, {})
    service_info = in_and_out.get(task.mid)
    
    if not service_info:
        return None
    
    local_inputs = [
        {
            "name": input_name,
            "value": inputs.get(input_name),
            "type": service_info["externalInput"].get(input_name)
        }
        for input_name in service_info["externalInput"].keys()
    ]
    
    local_outputs = [
        {
            "name": output_name,
            "value": outputs.get(output_name),
            "type": service_info["externalOutput"].get(output_name)
        }
        for output_name in service_info["externalOutput"].keys()
    ]
    
    return {
        "mid": task.mid,
        "taskId": task.id,
        "type": service_info["type"],
        "service": service_info["name"],
        "owner": task.owner,
        "inputs": local_inputs,
        "outputs": local_outputs,
        "end_time": task.end_time.isoformat() if task.end_time else None
    }

