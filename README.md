# PITS
**PITS: Variational Pitch Inference without Fundamental Frequency for End-to-End Pitch-controllable TTS**

**Abstract**: Previous pitch-controllable text-to-speech (TTS) models rely on directly modeling fundamental frequency, leading to low variance in synthesized speech. To address this issue, we propose PITS, an end-to-end pitch-controllable TTS model that utilizes variational inference to model pitch. Based on VITS, PITS incorporates the Yingram encoder, the Yingram decoder, and adversarial training of pitch-shifted synthesis to achieve pitch-controllability. Experiments demonstrate that PITS generates high-quality speech that is indistinguishable from ground truth speech and has high pitch-controllability without quality degradation. Code and audio samples will be available at https://github.com/anonymous-pits/pits.

**Training code is uploaded.**

**Demo and Checkpoint are uploaded at** [Hugging Face Space](https://huggingface.co/spaces/anonymous-pits/pits)🤗

Audio samples are uploaded at [github.io](https://anonymous-pits.github.io/pits/).

For the pitch-shifted Inference, we unify to use the notation in scope-shift, s, instead of pitch-shift.

Preprint version contains some errors! Please wait for the update!

![overall](asset/overall.png) 

## References
- Official VITS Implementation: https://github.com/jaywalnut310/vits
- NANSY Implementation from dhchoi99: https://github.com/dhchoi99/NANSY
- Official Avocodo Implementation: https://github.com/ncsoft/avocodo
- Official PhaseAug Implementation: https://github.com/mindslab-ai/phaseaug
- Tacotron Implementation from keithito: https://github.com/keithito/tacotron
- CSTR VCTK Corpus (version 0.92): https://datashare.ed.ac.uk/handle/10283/3443
- G2P for demo, g2p\_en from Kyubyong: https://github.com/Kyubyong/g2p
