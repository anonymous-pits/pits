import gradio as gr
import argparse
import torch
import commons
import utils
from models import (
    SynthesizerTrn, )

from text.symbols import symbol_len, lang_to_dict

# we use Kyubyong/g2p for demo instead of our internal g2p
# https://github.com/Kyubyong/g2p
from g2p_en import G2p
import re

_symbol_to_id = lang_to_dict("en_US")

class GradioApp:

    def __init__(self, args):
        self.hps = utils.get_hparams_from_file(args.config)
        self.device = "cpu"
        self.net_g = SynthesizerTrn(symbol_len(self.hps.data.languages),
                                    self.hps.data.filter_length // 2 + 1,
                                    self.hps.train.segment_size //
                                    self.hps.data.hop_length,
                                    midi_start=-5,
                                    midi_end=75,
                                    octave_range=24,
                                    n_speakers=len(self.hps.data.speakers),
                                    **self.hps.model).to(self.device)
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(args.checkpoint_path, model_g=self.net_g)
        self.g2p = G2p()
        self.interface = self._gradio_interface()

    def get_phoneme(self, text):
        phones = [re.sub("[0-9]", "", p) for p in self.g2p(text)]
        tone = [0 for p in phones]
        if self.hps.data.add_blank:
            text_norm = [_symbol_to_id[symbol] for symbol in phones]
            text_norm = commons.intersperse(text_norm, 0)
            tone = commons.intersperse(tone, 0)
        else:
            text_norm = phones
        text_norm = torch.LongTensor(text_norm)
        tone = torch.LongTensor(tone)
        return text_norm, tone, phones
    
    def inference(self, text, speaker_id_val, seed, scope_shift, duration):
        seed = int(seed)
        scope_shift = int(scope_shift)
        torch.manual_seed(seed)
        text_norm, tone, phones = self.get_phoneme(text)
        x_tst = text_norm.to(self.device).unsqueeze(0)
        t_tst = tone.to(self.device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([text_norm.size(0)]).to(self.device)
        speaker_id = torch.LongTensor([speaker_id_val]).to(self.device)
        decoder_inputs,*_ = self.net_g.infer_pre_decoder(
                                           x_tst,
                                           t_tst,
                                           x_tst_lengths,
                                           sid=speaker_id,
                                           noise_scale=0.667,
                                           noise_scale_w=0.8,
                                           length_scale=duration,
                                           scope_shift=scope_shift)
        audio = self.net_g.infer_decode_chunk(
            decoder_inputs, sid=speaker_id)[0, 0].data.cpu().float().numpy()
        del decoder_inputs,  
        return phones, (self.hps.data.sampling_rate, audio)


    def _gradio_interface(self):
        title = "PITS Demo"
        self.inputs = [
            gr.Textbox(label="Text (150 words limitation)",
                       value="This is demo page.",
                       elem_id="tts-input"),
            gr.Dropdown(list(self.hps.data.speakers),
                        value="p225",
                        label="Speaker Identity",
                        type="index"),
            gr.Slider(0, 65536, value=0, step=1, label="random seed"),
            gr.Slider(-15, 15, value=0, step=1, label="scope-shift"),
            gr.Slider(0.5, 2., value=1., step=0.1,
                      label="duration multiplier"),
        ]
        self.outputs = [
            gr.Textbox(label="Phonemes"),
            gr.Audio(type="numpy", label="Output audio")
        ]
        description = "Welcome to the Gradio demo for PITS: Variational Pitch Inference without Fundamental Frequency for End-to-End Pitch-controllable TTS.\n In this demo, we utilize an open-source G2P library (g2p_en) with stress removing, instead of our internal G2P.\n You can fix the latent z by controlling random seed.\n You can shift the pitch scope, but please note that this is opposite to pitch-shift. In addition, it is cropped from fixed z so please check pitch-controllability by comparing with normal synthesis.\n Thank you for trying out our PITS demo!"
        article = "Github:https://github.com/anonymous-pits/pits \n Our current preprint contains several errors. Please wait for next update."
        examples = [["This is a demo page of the PITS."],["I love hugging face."]]
        return gr.Interface(
            fn=self.inference,
            inputs=self.inputs,
            outputs=self.outputs,
            title=title,
            description=description,
            article=article,
            cache_examples=False,
            examples=examples,
        )

    def launch(self):
        return self.interface.launch(share=False)


def parsearg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="./configs/config_en.yaml",
                        help='Path to configuration file')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='PITS',
                        help='Model name')
    parser.add_argument('-r',
                        '--checkpoint_path',
                        type=str,
                        default='./logs/pits_vctk_AD_3000.pth',
                        help='Path to checkpoint for resume')
    parser.add_argument('-f',
                        '--force_resume',
                        type=str,
                        help='Path to checkpoint for force resume')
    parser.add_argument('-d',
                        '--dir',
                        type=str,
                        default='/DATA/audio/pits_samples',
                        help='root dir')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parsearg()
    app = GradioApp(args)
    app.launch()
