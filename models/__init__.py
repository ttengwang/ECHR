from models.OldModel_NEW import ShowAttendTellModel, AllImgModel, H3Model, TwostreamModel, Twostream_jump_Model, ThreestreamModel, TwostreamModel_3LSTM, H3denseModel, H3denaddModel, ThreestreamModel_2stream,ThreestreamModel_2stream_LDA,ThreestreamModel_2stream_CC
from models.sst_model import SST
from models.MA_attention_8_NEW import MA_Attention8


def setup_lm(lm_opt):
    if lm_opt.caption_model == 'show_attend_tell':
        model = ShowAttendTellModel(lm_opt)
    elif lm_opt.caption_model == 'three_stream':
        assert lm_opt.CG_num_layers ==3
        model = ThreestreamModel(lm_opt)
    return model


def setup_tap(tap_opt):
    if tap_opt.tap_model == 'SST':
        model = SST(tap_opt)
    else:
        raise Exception("tap model not supported: {}".format(tap_opt.tap_model))
    return model


def setup_fusion(fusion_opt):
    if fusion_opt.fusion_model == 'TSRM8':
        model = MA_Attention8(fusion_opt)
    else:
        raise Exception("fusion model not supported: {}".format(fusion_opt.fusion_model))

    return model
