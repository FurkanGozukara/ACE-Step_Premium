<h1>ACE-Step 1.5 XL Premium v3.1 - Better Music & Song Generator Than SUNO 5.0 - Remix and Repaint Features - Windows, RunPod, Massed Compute, Linux 1-click Installers</h1>

<h2>Download app : https://www.patreon.com/posts/157675060</h2>

<h2>Quick Info</h2>

**<p>Latest installer zip file : https://www.patreon.com/posts/157675060 </p>**

<p>ACE Step 1.5 XL is the newest State Of The Art (SOTA) Music and Song generator model. It has 3 variants and we support all 3 variants (Turbo, SFT, Base) with fully automatic setup, models download, VRAM presets for all GPUs starting from 4 GB and with all best researched generation values / settings / configurations.</p>

<img height="600" alt="image" src="https://github.com/user-attachments/assets/6766ac91-650f-4fa6-b0b4-3a23d85c42a5" />

- The zip file contains installers for
	- Windows : Windows_Install_or_Update.bat
		- Please follow requirements video for Windows before starting installation : https://youtu.be/DrhUHnYfwC0
		- Requirements tutorial is 1 time mandatory for all of my applications
  			- Windows installer will download only ACEStep 1.5 XL Turbo model
			- To download all models also run Windows_Download_All_Models.bat after installation
	- RunPod and SimplePod : Runpod_SimplePod_ACE_Step_Instructions.txt
	- Massed Compute / Local Linux : Massed_Compute_Instructions_READ.txt
 	- RunPod, Massed Compute installers automatically downloads all 3 ACEStep 1.5 XL models, Turbo, SFT and Base
  	- Zip file also has ACE_Step_Lyric_Generation_Instructions_For_LLMs.txt which you can use to better format your Music / Song lyrics and style by providing this file to your favorite LLM
  	- The installers will generate a Python 3.11 VENV automatically and install everything inside there, thus your system or any other of your APPs will never be impacted
  		- For Windows, you have to have Python 3.11 properly installed into your system, cloud images are all pre-setup by me so nothing needed just follow instructions
  	 	- With our pre-compiled libraries for both Windows and Linux, we support literally all of the GPUs out there starting from RTX 2000 series to RTX 5000 series, or cloud GPUs like RTX PRO 6000, H100, B200, etc.
  	  		- We have pre-compiled Flash Attention latest, xFormers latest, Sage Attention latest, Triton latest with Torch 2.9.1, CUDA 13
  	    	- Our installer uses uv installation for all of the requirements at once thus it is lightning fast, like 100x faster than normal

<img height="600" alt="2" src="https://github.com/user-attachments/assets/f2eed919-c0f0-42d9-b5b7-978f062a1824" />

<h3>24 May 2026 V3.1 Update</h3>

- When auto labeling, even though process finished it was still showing as processing at Raw Lyrics (from .txt file) and this bug fixed
- The training speed display fixed and now it shows training speed accurately after first 20 steps completed like below
- Just run installer to update

<img height="600" alt="image" src="https://github.com/user-attachments/assets/4e46ed9c-4bb1-4590-8998-329963a00f3b" />

<h3>24 May 2026 V3.0 Update</h3>

- V3 is an important update for LoRA training
- Some LoRA training bugs fixed and the training made more smoother
- I also have compiled Flash Attention 2.8.4 latest version for Windows and Linux with massive GPU support for Torch 2.11 and CUDA 13
- The reason is that we have upgraded our installer to Torch 2.11, CUDA 13 and Torchao 0.17.0 thus training is now even faster
- This Flash Attention compile costed me like 50$ on RunPod for Linux and over 14 hours on Windows you can read more info here : https://www.patreon.com/posts/159064759
- Download latest zip file, overwrite older files, delete ACE-Step_Premium\venv and run Windows_Install_or_Update.bat again to update or install

<h3>20 May 2026 V2.1 Update</h3>

- Please read V2.0 update first
- Browse Dataset JSON bug fixed which is needed to Preprocess files and generate training tensor files

<img height="600" alt="image" src="https://github.com/user-attachments/assets/b65b58ba-6ad4-427c-96fd-7ec33b836b5e" />

- LoRA refresh added to quick generation panel
  
  <img width="1318" height="474" alt="image" src="https://github.com/user-attachments/assets/240449ae-709e-4df6-8aad-a1cc7376904d" />
  
- In train LoRA tab we have improved the parameters you can set and default parameters are also improved
- Now when you change training base model, it will auto update training parameters to best for each model specifics
- Now Shift and Training Timestep Steps working accurately and set for each model : Base, SFT, Turbo
- Now Resume Training State directly takes saved state file, state files are now directly saved read below to see

<img height="600" alt="image" src="https://github.com/user-attachments/assets/62d1eaa4-e08d-44e6-a6df-ab70cc490820" />

- Unnecessary export LoRA and custom samples directory removed
- How training generated files saved completely revamped and improved
- Everything will be saved inside target folder with your training name like below 
- Much more organized and clean and ready to use after training
- safetensors files are LoRA files ready to use and pt files are state files which you can use to continue training
- Get latest zip file and run Windows_Install_or_Update.bat to update
	- Torchao upgraded to 0.16.0 for training
- Zip file changes only when needed

<img width="1121" height="585" alt="image" src="https://github.com/user-attachments/assets/05491eac-549b-4a08-9b9e-03156dd8fe39" />
<img width="1109" height="459" alt="image" src="https://github.com/user-attachments/assets/aa13220b-05ca-4e38-919a-e20ceaf1dd64" />



<h3>20 May 2026 V2.0 Update</h3>

- This is a massive upgrade and we have added so many new features so read carefully all
- The deault models were all 32-bit however we were generating songs in BF16 or FP8 / Int8
	- Thus, the models were keeping double size on disk for no reason and taking more RAM and duration to load
	- Therefore, I have generated BF16 models and updated the app and model downloader
	- Thus, you can make a complete fresh install or, delete \ACE-Step_Premium\models folder and run Windows_Install_or_Update.bat / Windows_Download_All_Models.bat again to download new models
	- Make sure to get latest zip file and overwrite previous installer files
	- New all 3 models folder takes 41.9 GB, previously it was 69.8 GB
- All default generation presets for all 3-models updated - the parameters are now more accurate 
	- unlimited (>24GB) , tier6b (20-24GB), tier6a (16-20GB), tier5 (12-16GB), tier4 (8-12GB), tier3 (6-8GB), tier2 (4-6GB), tier1 (≤4GB)
	- ACEStep XL 1.5 Turbo, ACEStep XL 1.5 SFT, ACEStep XL 1.5 Base
	- Thus, you may expect better quality generation on ACEStep Turbo, SFT and Base models
 - LoRA selection added to Generate song tab as well since we now fully support LoRA training
 - Also before starting anything with LoRA training, select your Model from here
 - If you gonna train ACEStep 1.5 XL SFT, first select it from this screen to load all best SFT parameters then continue this is important

<img height="600" alt="image" src="https://github.com/user-attachments/assets/d3b16956-3689-45a2-9e3e-79ce719cbcc4" />

 - Use isolated subprocess generation was not working properly and this is fixed
 - Cancel Generation button added to both Generate Song and Advanced tab
 	- For this button to work, Use isolated subprocess has to be enabled
- 🎓 LoRA Training tab completely remade
	- Now you can browse Dataset JSON Path and directly load
	- Now you can browse Browse Audio Folder and directly load
   
<img height="600" alt="image" src="https://github.com/user-attachments/assets/d3007cf2-fa50-4adb-a04d-f3326ad3a2b4" />
   
 - Dataset generation settings now fully working
	- Format Lyrics (LM) is not recommended
	- Transcribe Lyrics (LM) is recommended
	- If there are existing lyric files, Transcribe Lyrics (LM) will keep lyrics as it is and fill other data like Duration, Label, BPM, Key, Caption
	- Format is, audio_file_name.txt in same folder as audio
 	- Also select your Dataset Model according to the model you gonna train and Dataset VRAM preset - it will be also auto set according to your GPU during initial start
  	- If you get Out of Memory Error (OOM), move to 1 below VRAM Preset
    - Available auto-label and preprocess presets : 24 GB+ - quality, 12-16 GB, 10 GB+	
    
<img height="600" alt="image" src="https://github.com/user-attachments/assets/96b133bb-1c8f-4ea6-977d-feb07d2aa34e" />

 - Auto-Label All now fully working all bugs and errors fixed
	- Now use Browse Label Folder to pick auto labels saved folder
	- It will save labels after each labelling done
	- With using same folder, it will continue wherever left
   
<img height="600" alt="image" src="https://github.com/user-attachments/assets/0219cb98-ccb6-4e42-8cf4-2b716e6233ac" />

 - Now you can navigate between each data item and manually change / fix and save
 - Selecting song from above listing will also update this Preview & Edit selected song

<img height="600" alt="image" src="https://github.com/user-attachments/assets/47ab27ae-7427-42d9-b9bd-c53a446a1dad" />
<img height="600" alt="image" src="https://github.com/user-attachments/assets/d77e3c24-e1d7-4ba2-9367-222ff1bac414" />

 - After auto labelling done, save your dataset into any desired location with any name
 	- Now this auto labelling is fully VRAM optimized and auto unloads after completed and 100% free up VRAM and RAM

<img width="3508" height="504" alt="image" src="https://github.com/user-attachments/assets/6004d477-5717-402c-8ff4-3d65ef9fab2d" />

- After dataset json saved, load dataset json and preprocess and generate training pt files into desired target folder
	- These pt tensor files will be used to train the model it contains every information needed to train
	- Now this preprocess is fully optimized and auto unloads after completed and 100% free up VRAM and RAM

<img height="600" alt="image" src="https://github.com/user-attachments/assets/958fede4-5c7c-4556-9467-dbc868ce2cd0" />

- Once you are done with Dataset builder move to Train LoRA tab
	- Train LoKr not tested yet but Train LoRA fully working and fully tested and optimized
   
<img width="702" height="97" alt="image" src="https://github.com/user-attachments/assets/11142709-4a64-4e8b-b348-98ccfdbd7c7a" />

- Select folder of preprocessed tensors and load it
	- App will auto select VRAM Preset according to your GPU at initial load
 	- If you get Out of Memory Error (OOM), move to 1 below VRAM Preset
  	- Available presets : 24GB+, 16-24 GB, 12-16 GB, 10 GB+, 8-10 GB
	- 48 GB or above GPUs can try turning off Gradient checkpointing - not tested yet 
- Select your LoRA Base Model again - same as auto label and tensor generation
- Set your LoRA Training Name
	- Currently other parameters are set as best according to VRAM Preset

<img height="600" alt="image" src="https://github.com/user-attachments/assets/620ca310-5650-43a3-8f11-9cb31191c53b" />

- Learning rate and Training Epoch Count is in research
- So far I tested below settings on 50 songs dataset
- Set your Epoch count and Save Every N Epochs before starting training
- Batch size 1 and Gradient Accumulation steps 1 are best quality
- Set your Output directory as well where LoRA and training state files will be saved
	- This will be auto set to Loras folder so you can auto select from advanced tab Loras
	- Quick LoRA selection option added to fast generation tab as well
 	- So you will be able to select any use your LoRAs immediately from this folder 
- You can also resume from training state file

<img height="600" alt="image" src="https://github.com/user-attachments/assets/1da754e9-da6c-4a5a-868a-568b8eda506a" />
  
- Once all is set until this point you can start training
- If you want to generate samples during training, we support it as well
- So before starting training, make sure to enable sample and write your style and lyrics if you want

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/67fb5438-48a5-45e3-a474-ba06de3f60cd" />
<img  height="600" alt="image" src="https://github.com/user-attachments/assets/3dff1e1a-65a7-429d-ae77-6012b87fc153" />
<img  height="600" alt="image" src="https://github.com/user-attachments/assets/8ec7d7d2-cbf9-406f-9659-a65c1f19bab4" />

- To update download latest zip file, overwrite older files and run Windows_Install_or_Update.bat file
- ACEStep XL 1.5 SFT model training recommend but I am testing right now not concluded yet
- Hopefully will make a full training tutorial soon
- It takes around 100 minutes for RTX 5090 for 5000 steps with sample generation so pretty fast and best config uses 22.6 GB VRAM - so fits into RTX 4090 or 3090 as well
- Model variants comparison as below

<img width="2435" height="569" alt="image" src="https://github.com/user-attachments/assets/4c8a0277-b10c-4709-bcfb-5313e71fd4af" />


<h3>15 May 2026 V1.1 Update</h3>

- The behaviour of Songs, Number of songs to generate sequentially is fixed
- Will generate multiple songs in a loop
- Just run Windows_Install_or_Update.bat to update

<h2>How To Use ACEStep 1.5 XL SECourses Premium App and its features</h2>

- Main app interface screenshot

<img height="600" alt="0" src="https://github.com/user-attachments/assets/57185562-5b03-4790-a48e-a8b1760bd457" />

- All 3 models SFT, Base and Turbo are fully automatically supported with VRAM presets

- Each preset automatically updates best values when you change the selected ACEStep XL 1.5 model from selection

- Each preset change also updates necessary config automatically according to VRAM tier selection

- More VRAM tiers may have higher quality since using some different configs and models such as using acestep-5Hz-lm-4B vs 1.7B vs 0.6B, etc.

	- So for best quality, you may run the app on cloud services

	- 3 minutes song generation takes around 40 seconds on RTX 5090 with Turbo model, other models are also very close to this and all are really fast locally

<img width="405" height="294" alt="image" src="https://github.com/user-attachments/assets/7fc3e680-990c-4fb9-ac8f-1522f79f9562" />
<img width="378" height="600" alt="image" src="https://github.com/user-attachments/assets/6a8aee7c-7a49-4439-a840-2bc313ae329e" />

- We have high quality FP8 Cache feature as well - custom implemented
- On the first time, it will generate FP8 Scaled version of the used model, save it and use it when you next time use FP8 Scaled

<img width="488" height="507" alt="image" src="https://github.com/user-attachments/assets/c3761b0a-f462-40e6-ada0-63e9c19c06ec" />

- You can set most needed parameters regarding song / music generation on our specially designed easy generation screen like below
	- Change language of your song
	- Change Male / Female
	- Set Instrumental
	- Set song duration, -1 means auto
	- Set number of songs you want to generate
- You can also provide an image and if provided it will generate additional MP4 music video with generated audio file with desired video resolution while keeping your image aspect ratio
- All generations are fully saved with full metadata into outputs folder as sub folders
	- You can use open outputs folder to open it quickly

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/3aa1cb3f-c198-4359-ad51-e5f9b4bf0875" />
<img  height="600" alt="3" src="https://github.com/user-attachments/assets/4b878c6c-d2f8-4af9-8835-c6c494b05651" />

- For advanced users our advanced tab supports all the features these models have
	- They are Custom, Remix and Repaint

<img width="3549" height="440" alt="image" src="https://github.com/user-attachments/assets/2a6fe22c-98c9-416c-bc3a-c23643dcdc24" />

- Every feature has a detailed instructions and description so read everything to understand how app works

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/4c3363eb-1403-4064-ac47-602cb72402c2" />
<img  height="600" alt="image" src="https://github.com/user-attachments/assets/1f20043f-8786-43eb-b7ba-8876dfe9c5f0" />
<img  height="600" alt="image" src="https://github.com/user-attachments/assets/1a272134-e3a9-455b-94f7-681fd75664dc" />
<img  height="600" alt="image" src="https://github.com/user-attachments/assets/8e914e6a-5a0c-41bd-b47c-40dc4393a9e6" />

- You can set all the parameters individually but totally not needed since the presets we developed auto sets all of them and service is auto initialized so no need to click

<img height="600" alt="image" src="https://github.com/user-attachments/assets/76a3e783-33c8-43a8-b736-a9c3081c3983" />

- We support LoRA folder feature as well that you can pick and use automatically
- LoRA folder is ACE-Step_Premium\Loras - put your custom LoRAs here, app also supports LoRA training

<img height="600" alt="image" src="https://github.com/user-attachments/assets/eb628e84-1b89-4a0b-b07e-3aa3d5648e03" />

- More custom parameters that you can set if you want but all auto set to best with presets

<img height="600" alt="image" src="https://github.com/user-attachments/assets/2dad6879-900b-4003-91a3-8dd3904c4e7b" />
<img height="600" alt="image" src="https://github.com/user-attachments/assets/a5bd9ad4-e9ca-4c29-85c6-ddc8649f4f95" />

- You can make repaint as well

<img width="3502" height="512" alt="image" src="https://github.com/user-attachments/assets/12e62e1f-35b9-4b2b-91a0-684393e1741a" />

- We have a custom Library page where you can see all your generations with their metadata
- It is daily based filtered and very convenient to use

<img height="600" alt="image" src="https://github.com/user-attachments/assets/f7ad703c-5610-4a53-bd51-08f80080d82a" />

- We have results page for some custom fast and easy operations

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/c904082a-466f-4be2-9265-a15a6d4efd29" />

- We have custom preset system where you can save your config and load them later if you want
- Your last saved / used config will be auto remembered at next launch
- Delete this folder to return back to defaults : ACE-Step_Premium\premium_user_presets

<img height="600" alt="image" src="https://github.com/user-attachments/assets/4a3b7851-73f6-44d6-8c0f-0a6c13df2a17" />

- We have fully working LoRA training and I plan to make tutorial with it later hopefully
- But currently you can use LLMs like Codex or Cursor or Claude or Gemini, etc. to get their help or look online sources and other sources or just try and learn yourself to train

<img height="600" alt="image" src="https://github.com/user-attachments/assets/386cb12a-7d40-4919-bf61-ac40a0ef5df9" />

- We have fully working batch folder processing
- Name your song txt files as you wish which will contain lyrics and also make their style files with suffix _style.txt in same folder
- e.g. rap_song.txt and rap_song_style.txt, awesomesong.txt and awesomesong_style.txt and so on

<img height="600" alt="image" src="https://github.com/user-attachments/assets/87bd7bcf-c464-47e1-8061-04f70c499e69" />


<h1 align="center">Pushing the Boundaries of Open-Source Music Generation</h1>
<p align="center">
    <a href="https://acemusic.ai">ACEMusic</a> |
    <a href="https://ace-step.github.io/ace-step-v1.5.github.io/">Project</a> |
    <a href="https://huggingface.co/ACE-Step/Ace-Step1.5">Hugging Face</a> |
    <a href="https://modelscope.cn/models/ACE-Step/Ace-Step1.5">ModelScope</a> |
    <a href="https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5">Space Demo</a> |
    <a href="https://discord.gg/PeWDxrkdj7">Discord</a> |
    <a href="https://arxiv.org/abs/2602.00744">Technical Report</a> |
    <a href="https://github.com/ace-step/awesome-ace-step">Awesome ACE-Step</a>
</p>

<p align="center">
    <img src="./assets/organization_logos.png" height="80" alt="StepFun Logo" style="vertical-align: middle;">
    &nbsp;&nbsp;
    <a href="https://acemusic.ai">
        <img src="./assets/acemusic-logo.svg" height="57" alt="ACEMusic - Try ACE-Step Online" style="vertical-align: middle; position: relative; top: 2px;">
    </a>
</p>

## 📰 News

> 🎵 **Want a faster & more stable experience? Try [acemusic.ai](https://acemusic.ai) — 100% free!**

- **[2026-04-02] 🎉 ACE-Step 1.5 XL (4B DiT) Released!** — We introduce the XL series with a 4B-parameter DiT decoder for higher audio quality. Three variants available: [xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base), [xl-sft](https://huggingface.co/ACE-Step/acestep-v15-xl-sft), [xl-turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo). Requires ≥12GB VRAM (with offload), ≥20GB recommended. All LM models fully compatible. See [Model Zoo](#-model-zoo) for details.

## Table of Contents

- [📰 News](#-news)
- [✨ Features](#-features)
- [⚡ Quick Start](#-quick-start)
- [🚀 Launch Scripts](#-launch-scripts)
- [📚 Documentation](#-documentation)
- [📖 Tutorial](#-tutorial)
- [🏗️ Architecture](#️-architecture)
- [🦁 Model Zoo](#-model-zoo)
- [🔬 Benchmark](#-benchmark)

## 📝 Abstract
🚀 We present ACE-Step v1.5, a highly efficient open-source music foundation model that brings commercial-grade generation to consumer hardware. On commonly used evaluation metrics, ACE-Step v1.5 achieves quality beyond most commercial music models while remaining extremely fast—under 2 seconds per full song on an A100 and under 10 seconds on an RTX 3090. The model runs locally with less than 4GB of VRAM, and supports lightweight personalization: users can train a LoRA from just a few songs to capture their own style.

🌉 At its core lies a novel hybrid architecture where the Language Model (LM) functions as an omni-capable planner: it transforms simple user queries into comprehensive song blueprints—scaling from short loops to 10-minute compositions—while synthesizing metadata, lyrics, and captions via Chain-of-Thought to guide the Diffusion Transformer (DiT). ⚡ Uniquely, this alignment is achieved through intrinsic reinforcement learning relying solely on the model's internal mechanisms, thereby eliminating the biases inherent in external reward models or human preferences. 🎚️

🔮 Beyond standard synthesis, ACE-Step v1.5 unifies precise stylistic control with versatile editing capabilities—such as cover generation, repainting, and vocal-to-BGM conversion—while maintaining strict adherence to prompts across 50+ languages. This paves the way for powerful tools that seamlessly integrate into the creative workflows of music artists, producers, and content creators. 🎸


## ✨ Features

<p align="center">
    <img src="./assets/application_map.png" width="100%" alt="ACE-Step Framework">
</p>

### ⚡ Performance
- ✅ **Ultra-Fast Generation** — Under 2s per full song on A100, under 10s on RTX 3090 (0.5s to 10s on A100 depending on think mode & diffusion steps)
- ✅ **Flexible Duration** — Supports 10 seconds to 10 minutes (600s) audio generation
- ✅ **Batch Generation** — Generate up to 8 songs simultaneously

### 🎵 Generation Quality
- ✅ **Commercial-Grade Output** — Quality beyond most commercial music models (between Suno v4.5 and Suno v5)
- ✅ **Rich Style Support** — 1000+ instruments and styles with fine-grained timbre description
- ✅ **Multi-Language Lyrics** — Supports 50+ languages with lyrics prompt for structure & style control

### 🎛️ Versatility & Control

| Feature | Description |
|---------|-------------|
| ✅ Reference Audio Input | Use reference audio to guide generation style |
| ✅ Cover Generation | Create covers from existing audio |
| ✅ Repaint & Edit | Selective local audio editing and regeneration |
| ✅ Track Separation | Separate audio into individual stems |
| ✅ Multi-Track Generation | Add layers like Suno Studio's "Add Layer" feature |
| ✅ Vocal2BGM | Auto-generate accompaniment for vocal tracks |
| ✅ Metadata Control | Control duration, BPM, key/scale, time signature |
| ✅ Simple Mode | Generate full songs from simple descriptions |
| ✅ Query Rewriting | Auto LM expansion of tags and lyrics |
| ✅ Audio Understanding | Extract BPM, key/scale, time signature & caption from audio |
| ✅ LRC Generation | Auto-generate lyric timestamps for generated music |
| ✅ LoRA Training | One-click annotation & training in Gradio. 8 songs, 1 hour on 3090 (12GB VRAM) |
| ✅ Quality Scoring | Automatic quality assessment for generated audio |

## 🔔 Staying ahead
Star ACE-Step on GitHub and be instantly notified of new releases
![](assets/star.gif)

## 🤝 Partners

<p align="center">
    <a href="https://www.comfy.org/"><img src="https://registry.comfy.org/_next/static/media/logo_blue.9ac227d3.png" alt="ComfyUI" height="40" style="margin: 5px;"></a>
    <a href="https://zilliz.com/"><img src="https://avatars.githubusercontent.com/u/18416694" alt="Zilliz" height="40" style="margin: 5px;"></a>
    <a href="https://milvus.io/"><img src="https://miro.medium.com/v2/resize:fit:2400/1*-VEGyAgcIBD62XtZWavy8w.png" alt="Milvus" height="40" style="margin: 5px;"></a>
    <a href="https://zeabur.com/"><img src="https://zeabur.notion.site/image/attachment%3A43bc244b-9a2d-4b96-9646-8392aa6fc862%3Alogo-dark_1.svg?table=block&id=318a221c-948e-8056-b3c0-f9c39ce543ba&spaceId=ba37aeb9-0937-401d-aa41-ce1d3b6ff778&userId=&cache=v2" alt="Zeabur" height="40" width="40" style="margin: 5px;"></a>
    <a href="https://majiks.studio"><img src="https://raw.githubusercontent.com/Majiks-Studio/majiks-brand-kit/main/logos/app-icon/png/app-icon-128.png" alt="Majik's Music Studio" height="40" width="40" style="margin: 5px;"></a>
</p>

## ⚡ Quick Start

> 🎵 **Don't want to install locally? Try [acemusic.ai](https://acemusic.ai) — 100% free, no GPU required!**

> **Requirements:** Python 3.11-3.12, CUDA GPU recommended (also supports MPS / ROCm / Intel XPU / CPU)
> 
> **Note:** ROCm on Windows requires Python 3.12 (AMD officially provides Python 3.12 wheels only)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Clone & install
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync

# 3. Launch Gradio UI (models auto-download on first run)
uv run acestep

# Or launch REST API server
uv run acestep-api
```

Open http://localhost:7860 (Gradio) or http://localhost:8001 (API).

> 📦 **Windows users:** A [portable package](https://files.acemusic.ai/acemusic/win/ACE-Step-1.5.7z) with pre-installed dependencies is available. See [Installation Guide](./docs/en/INSTALL.md#-windows-portable-package).

> 📦 **MacOS users:** A [portable package](https://files.acemusic.ai/acemusic/mac/ACE-Step-1.5.zip) with pre-installed dependencies is available. See [Installation Guide](./docs/en/INSTALL.md#-macos-portable-package).

> 📖 **Full installation guide** (AMD/ROCm, Intel GPU, CPU, environment variables, command-line options): [English](./docs/en/INSTALL.md) | [中文](./docs/zh/INSTALL.md) | [日本語](./docs/ja/INSTALL.md)

### 💡 Which Model Should I Choose?

| Your GPU VRAM | Recommended DiT | Recommended LM Model | Backend | Notes |
|---------------|----------------|---------------------|---------|-------|
| **≤6GB** | 2B turbo | None (DiT only) | — | LM disabled by default; INT8 quantization + full CPU offload |
| **6-8GB** | 2B turbo | `acestep-5Hz-lm-0.6B` | `pt` | Lightweight LM with PyTorch backend |
| **8-16GB** | 2B turbo/sft | `acestep-5Hz-lm-0.6B` / `1.7B` | `vllm` | 0.6B for 8-12GB, 1.7B for 12-16GB |
| **16-20GB** | 2B sft or XL turbo | `acestep-5Hz-lm-1.7B` | `vllm` | XL requires CPU offload below 20GB |
| **20-24GB** | XL turbo/sft | `acestep-5Hz-lm-1.7B` | `vllm` | XL fits without offload; 4B LM available |
| **≥24GB** | XL sft (or xl-base for extract/lego/complete) | `acestep-5Hz-lm-4B` | `vllm` | Best quality, all models fit without offload |

> **XL (4B) models** (`acestep-v15-xl-*`) offer higher audio quality with ~9GB VRAM for weights (vs ~4.7GB for 2B). They require ≥12GB VRAM (with offload + quantization) or ≥20GB (without offload). All LM models are fully compatible with XL.

The UI automatically selects the best configuration for your GPU. All settings (LM model, backend, offloading, quantization) are tier-aware and pre-configured.

> 📖 GPU compatibility details: [English](./docs/en/GPU_COMPATIBILITY.md) | [中文](./docs/zh/GPU_COMPATIBILITY.md) | [日本語](./docs/ja/GPU_COMPATIBILITY.md) | [한국어](./docs/ko/GPU_COMPATIBILITY.md)

## 🚀 Launch Scripts

Ready-to-use launch scripts for all platforms with auto environment detection, update checking, and dependency installation.

| Platform | Scripts | Backend |
|----------|---------|---------|
| **Windows** | `start_gradio_ui.bat`, `start_api_server.bat` | CUDA |
| **Windows (ROCm)** | `start_gradio_ui_rocm.bat`, `start_api_server_rocm.bat` | AMD ROCm |
| **Linux** | `start_gradio_ui.sh`, `start_api_server.sh` | CUDA |
| **macOS** | `start_gradio_ui_macos.sh`, `start_api_server_macos.sh` | MLX (Apple Silicon) |

```bash
# Windows
start_gradio_ui.bat

# Linux
chmod +x start_gradio_ui.sh && ./start_gradio_ui.sh

# macOS (Apple Silicon)
chmod +x start_gradio_ui_macos.sh && ./start_gradio_ui_macos.sh
```

### ⚙️ Customizing Launch Settings

**Recommended:** Create a `.env` file to customize models, ports, and other settings. Your `.env` configuration will survive repository updates.

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred settings
# Examples in .env:
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
PORT=7860
LANGUAGE=en
```

> 📖 **Script configuration & customization:** [English](./docs/en/INSTALL.md#-launch-scripts) | [中文](./docs/zh/INSTALL.md#-启动脚本) | [日本語](./docs/ja/INSTALL.md#-起動スクリプト)

## 📚 Documentation

### Usage Guides

| Method | Description | Documentation |
|--------|-------------|---------------|
| 🖥️ **Gradio Web UI** | Interactive web interface for music generation | [Guide](./docs/en/GRADIO_GUIDE.md) |
| 🧭 **UI Support Baseline** | Supported UI boundary and future UI parity checklist | [Guide](./docs/en/UI_SUPPORT.md) |
| 🎛️ **VST3 Plugin** | Standalone VST3 plugin (C++/GGML) for DAW integration | [acestep.vst3](https://github.com/ace-step/acestep.vst3) |
| 🐍 **Python API** | Programmatic access for integration | [Guide](./docs/en/INFERENCE.md) |
| 🌐 **REST API** | HTTP-based async API for services | [Guide](./docs/en/API.md) |
| ⌨️ **CLI** | Interactive wizard and configuration | [Guide](./docs/en/CLI.md) |

### Setup & Configuration

| Topic | Documentation |
|-------|---------------|
| 📦 Installation (all platforms) | [English](./docs/en/INSTALL.md) \| [中文](./docs/zh/INSTALL.md) \| [日本語](./docs/ja/INSTALL.md) |
| 🎮 GPU Compatibility | [English](./docs/en/GPU_COMPATIBILITY.md) \| [中文](./docs/zh/GPU_COMPATIBILITY.md) \| [日本語](./docs/ja/GPU_COMPATIBILITY.md) |
| 🔧 GPU Troubleshooting | [English](./docs/en/GPU_TROUBLESHOOTING.md) |
| 🔬 Benchmark & Profiling | [English](./docs/en/BENCHMARK.md) \| [中文](./docs/zh/BENCHMARK.md) |

### Multi-Language Docs

| Language | API | Gradio | Inference | Tutorial | LoRA Training | Install | Benchmark |
|----------|-----|--------|-----------|----------|---------------|---------|-----------|
| 🇺🇸 English | [Link](./docs/en/API.md) | [Link](./docs/en/GRADIO_GUIDE.md) | [Link](./docs/en/INFERENCE.md) | [Link](./docs/en/Tutorial.md) | [Link](./docs/en/LoRA_Training_Tutorial.md) | [Link](./docs/en/INSTALL.md) | [Link](./docs/en/BENCHMARK.md) |
| 🇨🇳 中文 | [Link](./docs/zh/API.md) | [Link](./docs/zh/GRADIO_GUIDE.md) | [Link](./docs/zh/INFERENCE.md) | [Link](./docs/zh/Tutorial.md) | [Link](./docs/zh/LoRA_Training_Tutorial.md) | [Link](./docs/zh/INSTALL.md) | [Link](./docs/zh/BENCHMARK.md) |
| 🇯🇵 日本語 | [Link](./docs/ja/API.md) | [Link](./docs/ja/GRADIO_GUIDE.md) | [Link](./docs/ja/INFERENCE.md) | [Link](./docs/ja/Tutorial.md) | [Link](./docs/ja/LoRA_Training_Tutorial.md) | [Link](./docs/ja/INSTALL.md) | — |
| 🇰🇷 한국어 | [Link](./docs/ko/API.md) | [Link](./docs/ko/GRADIO_GUIDE.md) | [Link](./docs/ko/INFERENCE.md) | [Link](./docs/ko/Tutorial.md) | [Link](./docs/ko/LoRA_Training_Tutorial.md) | — | — |

## 📖 Tutorial

**🎯 Must Read:** Comprehensive guide to ACE-Step 1.5's design philosophy and usage methods.

| Language | Link |
|----------|------|
| 🇺🇸 English | [English Tutorial](./docs/en/Tutorial.md) |
| 🇨🇳 中文 | [中文教程](./docs/zh/Tutorial.md) |
| 🇯🇵 日本語 | [日本語チュートリアル](./docs/ja/Tutorial.md) |

This tutorial covers: mental models and design philosophy, model architecture and selection, input control (text and audio), inference hyperparameters, random factors and optimization strategies.

## 🔨 Train

📖 **LoRA Training Tutorial** — step-by-step guide covering data preparation, annotation, preprocessing, and training:

| Language | Link |
|----------|------|
| 🇺🇸 English | [LoRA Training Tutorial](./docs/en/LoRA_Training_Tutorial.md) |
| 🇨🇳 中文 | [LoRA 训练教程](./docs/zh/LoRA_Training_Tutorial.md) |
| 🇯🇵 日本語 | [LoRA トレーニングチュートリアル](./docs/ja/LoRA_Training_Tutorial.md) |
| 🇰🇷 한국어 | [LoRA 학습 튜토리얼](./docs/ko/LoRA_Training_Tutorial.md) |

See also the **LoRA Training** tab in Gradio UI for one-click training, or [Gradio Guide - LoRA Training](./docs/en/GRADIO_GUIDE.md#lora-training) for UI reference.

🔧 **Advanced Training with [Side-Step](https://github.com/koda-dernet/Side-Step)** — CLI-based training toolkit with corrected timestep sampling, LoKR adapters, VRAM optimization, gradient sensitivity analysis, and more. See the [Side-Step documentation](./docs/sidestep/Getting%20Started.md).

## 🏗️ Architecture

<p align="center">
    <img src="./assets/ACE-Step_framework.png" width="100%" alt="ACE-Step Framework">
</p>

## 🦁 Model Zoo

<p align="center">
    <img src="./assets/model_zoo.png" width="100%" alt="Model Zoo">
</p>

### DiT Models

| DiT Model | Pre-Training | SFT | RL | CFG | Step | Refer audio | Text2Music | Cover | Repaint | Extract | Lego | Complete | Quality | Diversity | Fine-Tunability | Hugging Face |
|-----------|:------------:|:---:|:--:|:---:|:----:|:-----------:|:----------:|:-----:|:-------:|:-------:|:----:|:--------:|:-------:|:---------:|:---------------:|--------------|
| `acestep-v15-base` | ✅ | ❌ | ❌ | ✅ | 50 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Medium | High | Easy | [Link](https://huggingface.co/ACE-Step/acestep-v15-base) |
| `acestep-v15-sft` | ✅ | ✅ | ❌ | ✅ | 50 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | High | Medium | Easy | [Link](https://huggingface.co/ACE-Step/acestep-v15-sft) |
| `acestep-v15-turbo` | ✅ | ✅ | ❌ | ❌ | 8 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Very High | Medium | Medium | [Link](https://huggingface.co/ACE-Step/Ace-Step1.5) |

### XL (4B) DiT Models

> XL models use a larger 4B-parameter DiT decoder (~9GB bf16) for higher audio quality. They require ≥12GB VRAM (with offload + quantization) or ≥20GB (without offload). All LM models are fully compatible.

| DiT Model | Pre-Training | SFT | RL | CFG | Step | Refer audio | Text2Music | Cover | Repaint | Extract | Lego | Complete | Quality | Diversity | Fine-Tunability | Hugging Face |
|-----------|:------------:|:---:|:--:|:---:|:----:|:-----------:|:----------:|:-----:|:-------:|:-------:|:----:|:--------:|:-------:|:---------:|:---------------:|--------------|
| `acestep-v15-xl-base` | ✅ | ❌ | ❌ | ✅ | 50 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | High | High | Easy | [Link](https://huggingface.co/ACE-Step/acestep-v15-xl-base) |
| `acestep-v15-xl-sft` | ✅ | ✅ | ❌ | ✅ | 50 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Very High | Medium | Easy | [Link](https://huggingface.co/ACE-Step/acestep-v15-xl-sft) |
| `acestep-v15-xl-turbo` | ✅ | ✅ | ❌ | ❌ | 8 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Very High | Medium | Medium | [Link](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo) |

### LM Models

| LM Model | Pretrain from | Pre-Training | SFT | RL | CoT metas | Query rewrite | Audio Understanding | Composition Capability | Copy Melody | Hugging Face |
|----------|---------------|:------------:|:---:|:--:|:---------:|:-------------:|:-------------------:|:----------------------:|:-----------:|--------------|
| `acestep-5Hz-lm-0.6B` | Qwen3-0.6B | ✅ | ✅ | ✅ | ✅ | ✅ | Medium | Medium | Weak | ✅ |
| `acestep-5Hz-lm-1.7B` | Qwen3-1.7B | ✅ | ✅ | ✅ | ✅ | ✅ | Medium | Medium | Medium | ✅ |
| `acestep-5Hz-lm-4B` | Qwen3-4B | ✅ | ✅ | ✅ | ✅ | ✅ | Strong | Strong | Strong | ✅ |

## 🔬 Benchmark

ACE-Step 1.5 includes `profile_inference.py`, a profiling & benchmarking tool that measures LLM, DiT, and VAE timing across devices and configurations.

```bash
python profile_inference.py                        # Single-run profile
python profile_inference.py --mode benchmark       # Configuration matrix
```

> 📖 **Full guide** (all modes, CLI options, output interpretation): [English](./docs/en/BENCHMARK.md) | [中文](./docs/zh/BENCHMARK.md)

## 📜 License & Disclaimer

This project is licensed under [MIT](./LICENSE)

ACE-Step enables original music generation across diverse genres, with applications in creative production, education, and entertainment. While designed to support positive and artistic use cases, we acknowledge potential risks such as unintentional copyright infringement due to stylistic similarity, inappropriate blending of cultural elements, and misuse for generating harmful content. To ensure responsible use, we encourage users to verify the originality of generated works, clearly disclose AI involvement, and obtain appropriate permissions when adapting protected styles or materials. By using ACE-Step, you agree to uphold these principles and respect artistic integrity, cultural diversity, and legal compliance. The authors are not responsible for any misuse of the model, including but not limited to copyright violations, cultural insensitivity, or the generation of harmful content.

🔔 Important Notice  
The only official website for the ACE-Step project is our GitHub Pages site.    
 We do not operate any other websites.  
🚫 Fake domains include but are not limited to:
ac\*\*p.com, a\*\*p.org, a\*\*\*c.org  
⚠️ Please be cautious. Do not visit, trust, or make payments on any of those sites.

## 🌐 Community & Ecosystem

Check out **[Awesome ACE-Step](https://github.com/ace-step/awesome-ace-step)** — a curated list of community projects, alternative UIs, ComfyUI nodes, cloud deployments, training tools, and more built around ACE-Step.

## 🙏 Acknowledgements

This project is co-led by ACE Studio and StepFun.


## 📖 Citation

If you find this project useful for your research, please consider citing:

```BibTeX
@misc{gong2026acestep,
	title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
	author={Junmin Gong, Yulin Song, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo}, 
	howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
	year={2026},
	note={GitHub repository}
}
```
