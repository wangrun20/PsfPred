from models.IKC import F_Model, P_Model, C_Model


def get_model(opt):
    match opt['name']:
        case None:
            return None
        case 'F_Model':
            return F_Model(opt)
        case 'P_Model':
            return P_Model(opt)
        case 'C_Model':
            return C_Model(opt)
        case _:
            raise NotImplementedError
