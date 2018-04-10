
def change_store_input(module, val):
    '''
    val should be True or False, the new value of store_input in supporting submodueles
    ''' 
    if getattr(module,"change_store_input",None) is not None:
        module.change_store_input(val)
    if getattr(module,"store_input",None) is not None:
        module.store_input=val
    try:
        for sublayer in module:
            sublayer.change_store_input(val)
        return
    except TypeError:
        return

def change_store_output(module, val):
    '''
    val should be True or False, the new value of store_output in supporting submodueles
    ''' 
    if getattr(module,"change_store_output",None) is not None:
        module.change_store_output(val)
    if getattr(module,"store_output",None) is not None:
        module.store_output=val
    try:
        for sublayer in module:
            sublayer.change_store_output(val)
        return
    except TypeError:
        return


