# Audio Style Transfer with Rhythmic Constraints

DAFx2018 "Audio Style Transfer with Rhythmic Constraints" code repository  
Maciek Tomczak  
Presented at http://dafx2018.web.ua.pt/  
Audio examples at https://maciek-tomczak.github.io/maciek.github.io/Audio-Style-Transfer-with-Rhythmic-Constraints

# Overview
We present a rhythmically constrained audio style transfer technique for automatic mixing and mashing of two audio inputs. In this transformation the rhythmic and timbral features of both input signals are combined together through the use of an audio style transfer process that transforms the files so that they adhere to a larger metrical structure of the chosen input. New loss function based on cosine similarity of rhythmic envelopes can be used with an addition of a third audio input for more varied transformations.

## Installation
### Required Packages
* [numpy](https://www.numpy.org)
* [scipy](https://www.scipy.org) 
* [madmom](https://github.com/CPJKU/madmom)   
* [librosa](https://librosa.github.io)
* [tensorflow](https://www.tensorflow.org)  

## Usage
### Command line
Mashup of two recordings where only the style loss is used:
```Python
audio_style_transfer.py -A inputA.wav -B inputB.wav
```
Transform three inputs together:
```Python
audio_style_transfer.py -A inputA.wav -B inputB.wav -C inputC.wav -pA style 0.3 -pB style 0.3 -pC content 0.4
```
Add loss based on cosine distance between target rhythmic envelopes:
```Python
audio_style_transfer.py -A inputA.wav -B inputB.wav -C inputC.wav -pA style 0.3 -pB style 0.3 -pC content 0.4 -mode ODF -target_odf_pattern C
```

## References
| **[1]** |                  **[Tomczak, M., C., Southall, J., Hockman, Audio Style Transfer with Rhythmic Constraints,                    Proceedings of the 21st International Conference on Digital Audio Effects (DAFx-18), Aveiro, Portugal, September 4–8, 2018.](https://github.com/maciek-tomczak/audio-style-transfer-with-rhythmic-constraints/blob/master/paper.pdf)**|
| :---- | :--- |
