




def load_catalog_for_fov(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load artifact created by save_catalog_for_fov.
    Returns dict with: incidence (bool), comp_labels (list), roi_index_order (np.ndarray), meta (dict)
    """
    from pathlib import Path
    import json
    path = Path(path)
    if path.suffix.lower() == '.npz':
        d = np.load(path, allow_pickle=True)
        inc = d['incidence'].astype(bool)
        labels = [str(x) for x in d['comp_labels']]
        roi_order = d['roi_index_order'].astype(int)
        meta = json.loads(str(d['meta_json']))
    else:
        payload = json.loads(path.read_text())
        inc = np.asarray(payload['incidence'], dtype=bool)
        labels = payload['comp_labels']
        roi_order = np.asarray(payload['roi_index_order'], dtype=int)
        meta = payload.get('meta', {})
    return dict(incidence=inc,
                comp_labels=labels,
                roi_index_order=roi_order,
                meta=meta)