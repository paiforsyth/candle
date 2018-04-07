def reset_masks(module):
    if getattr(module,"reset_masks",None) is not None:
        module.reset_masks()
        return
    try:
        for sublayer in module:
            reset_masks(sublayer)
        return
    except TypeError:
        return

