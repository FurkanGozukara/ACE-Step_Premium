<h1>ACE-Step 1.5 XL Premium - Better Music & Song Generator Than SUNO 5.0 - Remix and Repaint Features - Windows, RunPod, Massed Compute, Linux 1-click Installers</h1>

<h2>Download app : https://www.patreon.com/posts/157675060</h2>

<h2>Quick Info</h2>

**<p>Latest installer zip file : https://www.patreon.com/posts/157675060 </p>**

## Quick Info

- This app has the following repos perfectly combined into our premium app with additional improvements and features such as optimized model loading, VRAM, quality, accuracy and performance optimizations, batch folder processing and many more (all models automatically downloaded and everything installed into a Python 3.11 VENV)        
    -   We got VRAM presets for every GPUs already set, read changelogs below to learn everything, slowly top to bottom read recommended        
        -   ACESTEP XL 1.5 (both inference + training) : [https://github.com/Runware/ACE-Step-1.5-XL](https://github.com/Runware/ACE-Step-1.5-XL)            
            -   [https://deepwiki.com/ace-step/ACE-Step-1.5/5-generation-features](https://deepwiki.com/ace-step/ACE-Step-1.5/5-generation-features)                
        -   SAM-Audio Segment from Facebook / META : [https://github.com/facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)            
            -   Massive optimizations made for this model , it is working amazing                
        -   Auto-Editor : [https://github.com/wyattblue/auto-editor](https://github.com/wyattblue/auto-editor)            
        -   TrackAICleaner Post Processing : [https://github.com/mikecastrodemaria/TrackAICleaner](https://github.com/mikecastrodemaria/TrackAICleaner)            
        -   DiffPitcher : [https://github.com/haidog-yaqub/DiffPitcher](https://github.com/haidog-yaqub/DiffPitcher)            
    
    **ACE Step 1.5 XL** is the newest State Of The Art (SOTA) Music and Song generator model. It has 3 variants and we support all 3 variants (Turbo, SFT, Base) with fully automatic setup, models download, VRAM presets for all GPUs starting from 4 GB and with all best researched generation values / settings / configurations.

<img height="600" alt="image" src="https://github.com/user-attachments/assets/6766ac91-650f-4fa6-b0b4-3a23d85c42a5" />

-   **Windows Requirements**    
    -   Python 3.11.x, FFmpeg, CUDA 13, cuDNN 9.17 or above, C++ tools, MSVC and Git        
        -   Don't worry CUDA 13 works with all GPUs - make sure you have updated NVIDIA driver            
        -   Follow this requirements tutorial video exactly : [https://youtu.be/DrhUHnYfwC0](https://youtu.be/DrhUHnYfwC0)            
        -   Follow its updated post with links and screenshots exactly : [https://www.patreon.com/posts/click-to-open-post-used-in-tutorial-111553210](https://www.patreon.com/posts/click-to-open-post-used-in-tutorial-111553210)
            
-   The zip file contains installers for    
    -   Windows : Windows\_Install\_or\_Update.bat        
        -   Please follow requirements video for Windows before starting installation : [https://youtu.be/DrhUHnYfwC0](https://youtu.be/DrhUHnYfwC0)            
        -   Requirements tutorial is 1 time mandatory for all of my applications            
        -   Windows installer will download only ACEStep 1.5 XL Turbo model            
            -   To download all models also run Windows\_Download\_All\_Models.bat after installation                
    -   RunPod and SimplePod : Runpod\_SimplePod\_ACE\_Step\_Instructions.txt        
    -   Massed Compute / Local Linux : Massed\_Compute\_Instructions\_READ.txt        
    -   RunPod, Massed Compute installers automatically downloads all 3 ACEStep 1.5 XL models, Turbo, SFT and Base        
    -   Zip file also has ACE\_Step\_Lyric\_Generation\_Instructions\_For\_LLMs.txt which you can use to better format your Music / Song lyrics and style by providing this file to your favorite LLM        
    -   The installers will generate a Python 3.11 VENV automatically and install everything inside there, thus your system or any other of your APPs will never be impacted        
        -   For Windows, you have to have Python 3.11 properly installed into your system, cloud images are all pre-setup by me so nothing needed just follow instructions            
        -   With our pre-compiled libraries for both Windows and Linux, we support literally all of the GPUs out there starting from RTX 2000 series to RTX 5000 series, or cloud GPUs like RTX PRO 6000, H100, B200, etc.            
            -   We have pre-compiled Flash Attention latest, xFormers latest, Sage Attention latest, Triton latest with Torch 2.11, CUDA 13                
            -   Our installer uses uv installation for all of the requirements at once thus it is lightning fast, like 100x faster than normal

<img height="600" alt="2" src="https://github.com/user-attachments/assets/f2eed919-c0f0-42d9-b5b7-978f062a1824" />

### 21 June 2026 V6.0 Update
-   **Gangam Style in English :** [**https://x.com/SECourses/status/2068512611725975733**](https://x.com/SECourses/status/2068512611725975733)    
-   This is a massive improvements and fixes update    
-   We have moved to the Gradio 6.19 and thus transformers library upgraded to 5.3    
    -   So i had to fix pipeline for newest transformers library        
    -   PyTorch 5Hz LM generation now uses modern Transformers forward-pass features such as cache\_position and logits\_to\_keep when available.        
-   Gradio interface event handling was heavily improved for Gradio 6.19.    
    -   Many UI sync events now run without queue overhead and without unnecessary progress overlays.        
    -   Stale Gradio status timers are automatically hidden, fixing stuck timer/progress artifacts.        
    -   Long-running actions now show progress only on relevant outputs instead of slowing down unrelated UI elements.        
    -   Library metadata display was changed from heavy JSON rendering to copy-friendly text, making the library page smoother.        
-   Batch processing, batch extract, audio processing, generation, library, LoRA, LoKR, dataset, and SAM Audio UI wiring were updated for smoother behavior.    
-   Model loading architecture updated for newer transformers library and now model loading faster    
    -   Due to newer transformers library, now torch compile is even faster than before        
-   PyTorch LM loading now tries the faster SDPA attention path on CUDA.    
-   Audio-code generation now has a compact valid-token sampling path, avoiding unnecessary full-vocabulary processing during constrained generation - No quality loss    
-   VAE tiled decode was optimized by preventing pathological tiny-stride chunking - Nno quality loss    
-   With VAE optimization + transformers library, now torch compile is able to generate full song in 30 seconds on RTX 5090    
-   GPU VRAM presets are re-tested and updated as below    

<img width="675" height="516" alt="image" src="https://github.com/user-attachments/assets/9bde2193-cd02-4431-8970-8d64f913775d" />

-   Wildcards special character was \[\] and now it is fixed and changed into {}

<img width="2166" height="204" alt="image" src="https://github.com/user-attachments/assets/115ce41f-c38b-4261-880a-64f9a868692b" />

-   CoT Language Detection and Caption / Style Auto Improve was enabled in some presets and now they are all disabled - so you have to manually enable    
    -   They were causing unexpected issues and problems        
    -   If CoT language is enabled, the LM-detected language is used only when vocal language is set to auto/unknown.        
        -   Explicit user-selected vocal language is preserved and no longer unexpectedly overwritten.            
-   In advanced tab now you can set explicit vocal language and this fixed so many issues    

<img width="1220" height="480" alt="image" src="https://github.com/user-attachments/assets/e7086cb8-bde4-4c6f-965f-9997ab7a452d" />

-   Advanced tab set Vocal Language will update Generate Song tab Vocal Language as well or vice-versa    
-   Remix presets implemented and literally 1-click first test result you can see here    
-   **Gangam Style in English :** [**https://x.com/SECourses/status/2068512611725975733**](https://x.com/SECourses/status/2068512611725975733)    
-   Hopefully will make a mini tutorial video so open bell on Youtube : [https://www.youtube.com/SECourses](https://www.youtube.com/SECourses)    

<img width="2119" height="551" alt="image" src="https://github.com/user-attachments/assets/fb5f1a12-6617-4e9a-a039-0c93b9e36ef4" />

-   Batch audio processing now reports status immediately when scanning starts.    
-   Batch Extract now normalizes itself to Extract mode internally, instead of requiring the user to manually switch generation mode first.    
-   Batch progress display was improved.    
-   Batch queue restore defaults now keep CoT caption/language disabled unless explicitly enabled.    
-   For updating please get latest v6 zip file, overwrite previous files and run installer bat file    
    -   If you get any errors, please delete ACE-Step\_Premium\\venv and then run installer again

### 18 June 2026 V5.5 Update

-   Full tutorial video published finally for inference : https://youtu.be/9C_6qNKjgpA
-   I started working on LoRA training tutorial as well hopefully soon
-   With 5.5 optimizer specific parameters are now shown that you can set, I am also working on to make them auto default best hopefully
-   There was a visual bug that hidden Remix Melody Retention and Direct Source Latents (no_fsq) on Remix songs page and this bug fixed and app scanned entirely and all visuals verified
	-   Default value set to 0.97 one of our expert remixer recommended that
-   Just run Windows_Install_or_Update.bat to update, the zip file not changed

<img width="3502" height="405" alt="image" src="https://github.com/user-attachments/assets/a8f192df-258d-4736-8ea1-fecac8448e2a" />

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/2212f840-2e78-4435-aa79-fd934db754aa" />

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/219f8633-cda3-4ffc-8a72-f84af11485b2" />

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/5b1aac4d-0732-44d0-8066-0ebcab962f56" />


### 18 June 2026 V5.4 Update

-   Now batch folder processing for ACESTEP XL 1.5 and SAM Audio has this extra option Save only output
	-   This is useful to get only processed files and no other stuff like remaining part of the songs or metadata files, etc.

<img width="3495" height="623" alt="image" src="https://github.com/user-attachments/assets/991cdec5-a61e-404e-8d86-d1bc061e8575" />

### 18 June 2026 V5.3 Update

-   Wildcard feature implemented
-   It works both for Style / Captions and Lyrics with syntax verification as well
-   Wildcard syntax is `{option 1|option 2|option 3}`; square-bracket lyric tags such as `[Verse]` stay unchanged
-   It will work in batch folder processing as well so you can write that way in txt files
	-   If you enable Auto improve lyrics or Auto improve style they may break your syntax so don't enable when using wildcards 
-   Just run Windows_Install_or_Update.bat to update same zip file still
-   Also full inference tutorial published that covers every topic in details including how to install on Windows, RunPod, Massed Compute and SimplePod : https://youtu.be/9C_6qNKjgpA

<img height="600" alt="image" src="https://github.com/user-attachments/assets/e359d264-3dac-4547-8453-8c7a89ee257d" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/622e8d27-93fe-4a8a-b91f-f07fb08f82f5" />


### 16 June 2026 V5.2 Update

-   Default Remix value is now 0.95 instead of 1
-   Seed box and Random seed option moved to a much easier to use place
-   Last generation seed value will be auto set in seedbox so you can uncheck random seed and keep working with same seed now easier

<img height="600" alt="image" src="https://github.com/user-attachments/assets/5f4e6c10-95aa-4a8c-b565-28dd85ff74da" />

### 14 June 2026 V5.1 Update 

-   Use Repaint with lyrics added to the Repaint tab of ACESTEP XL 1.5

<img width="1097" height="247" alt="image" src="https://github.com/user-attachments/assets/4a8dc592-95eb-43ac-abed-00e4ab88660f" />

### 14 June 2026 V5.0.0 Update 

-   I am still working on inference tutorial and as I used and as you made new feautre requests new features arrived    
-   Auto-Editor trim output was not working properly and this bug fixed now should work much better when you use it in SAM Audio Segment or ACESTEP XL 1.5 Extract    
    -   This is really useful to get only vocals and trim empty / no vocal parts for training        
-   Extract All stems feature implemented to ACESTEP XL 1.5 extract tab
    
<img width="2026" height="202" alt="image" src="https://github.com/user-attachments/assets/e400fca0-cd85-4ddd-9d7d-d1bfbb5616be" />
    
-   Extracted stems will be saved in same folder with suffixes like brass, guitar, vocal, etc.    
    -   I noticed that extracting stems much better working on full songs rather than part of songs like 1 minute split part for some reason for ACESTEP XL 1.5 extract        
    -   Extract logic improved        
    -   Each different extract may yield different results so you can try multiple times to get better extract
        
-   Auto-Editor workflow export significantly improved    
    -   In Audi Processing tab enable Auto-Editor trim silent sections        
    -   Then Set Processing Preset = None        
    -   Then select your Auto-Editor workflow export like DaVinci Resolve        
    -   Then use Local Audio/Video Path with Browse File button or direct path        
    -   This way you will get almost instantly .fcpxml with accurate file path or whatever supported format you pick        

<img height="600" alt="image" src="https://github.com/user-attachments/assets/7c10a812-c34d-4b01-8830-b212d963c31e" />

-   SAM Audio Segmet now supports Batch Segment    
-   You can use Batch Segment with 2 ways    
    -   First way is enable Batch Segment checkbox and type your stems / segments into Custom Prompt with ; seperation        
    -   Second way is select multiple Quick Prompt from dropdown and it will segment / extract every one of them        
    -   Custom prompt section overwrites Quick Prompt selections        
    -   Extracted stems / segments will be saved in same folder with suffixes like brass, guitar, vocal, etc.        

<img height="600" alt="image" src="https://github.com/user-attachments/assets/fe116d0c-6a46-4515-a5a3-8771a8faf149" />

-   Load Metadata feature implemented as a new tab    
-   Select the generation\_manifest.json and it will load every single configuration / parameter of that generation    

<img height="600" alt="image" src="https://github.com/user-attachments/assets/2f31f454-3a15-4e36-a212-8946d60bedcf" />

-   Get the latest zip file, overwrite older files and run Windows\_Install\_or\_Update.bat file for update or fresh install    
-   To have all models (ACESTEP XL 1.5 Base and ACESTEP XL 1.5 SFT) run Windows\_Download\_All\_Models.bat after installation

### 12 June 2026 V4.9.5 Update 

-   In Audio Processing tab now there is None Processing Preset which unchecks all Audio Enhancement    
-   Now there is Disable upload preview checkbox in Audio Processing tab    
    -   Use for very large videos or containers like multi-GB MKV files. When enabled, Gradio will not render the uploaded media preview, avoiding slow browser/Gradio post-processing such as MKV-to-MP4 preview conversion. Processing still uses the original uploaded file.        
    -   Gradio does post processing to every video file if not mp4 therefore other formats will take massive time to display if they are big : [https://github.com/gradio-app/gradio/issues/13527](https://github.com/gradio-app/gradio/issues/13527)

### 11 June 2026 V4.9.4 Update 

-   Auto-Editor trim silent parts descriptions updated    

<img width="3540" height="608" alt="image" src="https://github.com/user-attachments/assets/2e14efb4-06ec-40ec-9e16-7169f76b6931" />

-   Analyze button won't overwrite your lyrics anymore    
-   Auto-Editor trim silent parts feature in SAM Audio Segment and ACESTEP XL 1.5 Extract will now use the settings / parameters set in Audio Processing tab    
-   In ACESTEP XL 1.5 extract mode when Auto-Editor trim was selected, it was not working accurately and now will work a bug fixed    

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/f129ea9e-cd89-4991-a55b-d4e1726f5fc0" />

-   Latest generated results sections labels fixed - for ACESTEP XL 1.5 advanced tab    

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/4fd58b1d-ac10-4f2b-8f7a-887d0a50ce2e" />

-   For update / install use latest zip file (4\_7), overwrite and run Windows\_Install\_or\_Update.bat

### 11 June 2026 V4.9.3 Update

-   In the repaint task, if generated song is shorter than the selected repaint area, it will trim thus you won't have silent parts    
-   The Repaint Strength description updated and fixed : When lyrics are provided, Repaint switches to text-to-music, so Repaint Strength has no impact when changing lyrics. To keep the same vocal audio, LoRA training and using that LoRA are mandatory.
    -   Now output format can be selected in ACESTEP XL 1.5 modes    
    -   Default is set as mp3 since generated files were taking too much space        
    -   Now all generated files will obey the selected format e.g. like below        

<img height="600" alt="image" src="https://github.com/user-attachments/assets/6f03c40c-2704-43ab-9002-32f8e7dd0214" />

-   Lego mode was not working accurately and this issue fixed    
-   Now in Lego mode, you will see only generated output as well such as you selected guitar so you will get the generated guitar song as well like below    

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/4e4c05c2-54bd-42c3-a818-1484b866a3d7" />

-   For update / install use latest zip file (4\_7), overwrite and run Windows\_Install\_or\_Update.bat

### 10 June 2026 V4.9 Update 

-   V4.9 is a pretty big and important update lots of fixes and improvements    
-   Generation modes now explicity shows recommended models for ACESTEP XL 1.5    

<img height="600" alt="image" src="https://github.com/user-attachments/assets/5ab0217d-37bd-4a51-a53d-7620b713bf75" />

-   Previously, switching models without restarting the app was causing VRAM leak and OOM    
    -   This issue is fixed and now you can generate with Turbo model and then switch SFT or Base, and so on        
        -   To be 100% sure not have any RAM or VRAM leak, enable Use isolated subprocess generation checkbox
        -   This option will slow subsequent generations and not mandatory, so enable if you are sure and needed       

<img width="727" height="202" alt="image" src="https://github.com/user-attachments/assets/ff8fc2e2-8029-4c04-a1af-e1ba9dd343bc" />

-   For Remix, Repaint, Lego and Complete, now you can set Instrument Start and End of source input and it will show live preview, really useful for Repaint    

<img width="3476" height="435" alt="image" src="https://github.com/user-attachments/assets/2947745b-725b-48ab-9f06-a90a5b643e9b" />

-   Instrument Start and End selection was not working accurately for Remix, Repaint, Lego and Complete but this bug fixed so now you can repaint just specific part of the model    
-   Repaint was not using accurate methodologies and automatic inner prompt to repaint song accurately and now this issue also fixed    
    -   So now you can change specific part of the song and make it sing different vocal / lyrics etc perfectly working tested        
-   Remix, Lego, Repaint and Complete mode errors fixed and they are made more robust    
-   Optional Parameters, Batch Process, Settings will be closed by default now, so easier to read interface    
    -   Click them to open them again        

<img height="600" alt="image" src="https://github.com/user-attachments/assets/9c41ffba-86e8-4357-8f82-24e079716409" />

-   Cluttering unrelated some information from Remix, Lego, Repaint and Complete modes removed such as Custom Guide from Remix    
-   Generated results now will show followings    
    -   Latest Generated Result (Sample 1) : Is the full new repainted, remixed, etc song        
    -   Next to it Original Input, the original song for quickly listen both and compare        
    -   Latest Repainted Area, is the area of the song you repainted like between 30-40 seconds, this works for other modes too so you can listen only that particular section        
    -   Next to it, Latest Repainted Area Original, the original part of the song that was repainted, etc. to see before after quickly        

<img height="600" alt="image" src="https://github.com/user-attachments/assets/aec6881d-8888-4515-b8dd-c0f590940422" />

-   For update / install use latest zip file (4\_7), overwrite and run Windows\_Install\_or\_Update.bat

### 10 June 2026 V4.8 Update 

<img width="1646" height="88" alt="image" src="https://github.com/user-attachments/assets/c899ae1a-9d73-4448-a603-963e3aed4385" />

-   Torch compile feature implemented for ACESTEP 1.5 XL and SAM Audio processing    
    -   For ACESTEP XL 1.5, switch to advanced tab and enable, then you can switch back to Generate Song tab        
    -   ACESTEP XL 1.5 training also supports torch compile but not tested and verified yet        
-   The initial torch compile may take some time but after that, repeated usage brings massive performance boost as shown as below    
    -   It won't recompile once compiled even if app is restarted, so it uses compile cache, if necessary it will recompile though        
    -   Initial compile may take time and may look like frozen but both inference and training tested and working        
-   You have to have accurately setup CUDA, MSVC and C++ Tools for this to work since Torch compile depends on it    
    -   Therefore, follow requirements tutorial fully properly : [https://youtu.be/DrhUHnYfwC0](https://youtu.be/DrhUHnYfwC0)        
-   The system is very robustly designed to automatically find accurate CUDA and C++ tools installation even if you have multiple installations    
-   LoRA training speed with Torch Compile is 0.98 it / second and without Torch Compile is 0.78 it / second    
    -   25% faster        
-   Use latest zip file, overwrite and run Windows\_Install\_or\_Update.bat to update    
-   ACESTEP XL 1.5 Inference Torch Compile    

<img width="1451" height="366" alt="image" src="https://github.com/user-attachments/assets/f75613fb-3440-4ea7-8301-7ddf0c3d34ae" />

-   SAM Audio Inference Torch Compile    

<img width="1503" height="209" alt="image" src="https://github.com/user-attachments/assets/67e11a61-8d91-4cc9-a609-58e8e1cf2cae" />

-   ACESTEP 1.5 XL LoRA Training Torch Compile

<img width="1866" height="472" alt="image" src="https://github.com/user-attachments/assets/aa46b51c-4e96-424c-9cd0-794750c9748b" />

<img width="2823" height="442" alt="image" src="https://github.com/user-attachments/assets/9c55b715-4994-47ae-b6b5-ae0b7210b0c1" />

### 6 June 2026 V4.7.1 Update

-   Auto-Editor executable download now has alternative source if GitHub fails - now more robust    
-   New feature DiffPitcher Pitch Fixer implemented into Audio Processing tab since requested    
    -   You can read more about it here : [https://github.com/haidog-yaqub/DiffPitcher](https://github.com/haidog-yaqub/DiffPitcher)        
    -   The installer bat file will download necessary diffusion models automatically as safetensors files        
-   Get latest zip file (4\_7), overwrite previous files and run Windows\_Install\_or\_Update.bat to update

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/9598bd46-ae11-40fc-a011-ce9de5388af5" />

### 6 June 2026 V4.6 Update

-   Audio processing tab significantly improved a lots of new features added  
-   Now supports Run as subprocess and cancel button immediately    
-   Now fully supports video inputs    
-   Now supports Export Only Audio - very useful for getting audio from video if you don't need video    
-   Now avoids reencoding of videos only if audio of video is processed - Auto-Editor triggers video processing    
-   Now supports video re-encoding profiles    
-   Now supports Auto-Editor workflow export for Davinci Resolve, Adobe Premiere Pro, Final Cut Pro, Shotcut and Kdenlive    
    -   Thus you can trim silent parts of your videos and continue editing in your favorite app, I use this literally to edit my tutorial videos        
-   Now fully shows Audio Processing tab process progress in CMD and also on Gradio    
    -   Auto-Editor video processing may take quite time since it re-encodes video        
-   Zip file is same, just use Windows\_Install\_or\_Update.bat to update

  <img height="600" alt="image" src="https://github.com/user-attachments/assets/87eb80a8-9e07-4f7b-84db-f077cd775dc6" />

### 4 June 2026 V4.5 Update

-   SAM Audio model loading speed significantly improved like 2.5x faster than before    
-   Unchecking Subprocess mode in SAM Audio was not working now works    
    -   So if you uncheck, after processing, it will keep model in VRAM thus instantly starts processing next task - in batch mode it doesn't unload model even if it is checked until batch process ends
-  New feature Predict spans added to the SAM Audio
	-  Uses SAM-Audio's span predictor to estimate target time ranges from the text prompt when you did not provide anchors
	-  This can improve quality of results depending on source file and the task so you can compare and see if improves
	-  This can use slightly more VRAM and slightly slower      
-   Advanced tab renamed into ACESTEP Advanced    
-   Interface of following sections Custom, Remix, Repaint, Extract,Lego, Complete improved which are located in ACESTEP Advanced tab    
-   **Descriptions and buggy features of each section updated and improved as below:**    
-   **Custom**: Manual mode for precise control over caption, lyrics, BPM, key, duration, sampler settings, and advanced generation parameters. Use it when you already have a clear target and want to tune the result yourself. Switch to Generate Song main tab when you want to describe the idea in plain language and let AI fill in the details.    
    -   **What it does:** generates new music from your manual Caption, Lyrics, BPM, key, duration, and advanced settings.        
    -   **How to use it:** describe the target style and vocal delivery in Caption, write structured Lyrics with tags such as \[Verse\] and \[Chorus\], then set metadata only when you need tighter control. Leave Think on when you want the LM to plan; turn Think off only when using pasted LM Codes Hints.        
    -   **Audio inputs:** Reference Audio can guide timbre, mix, performance feel, and atmosphere, but it will not copy exact melody, rhythm, or lyrics. Source Audio is ignored in normal Custom generation and is only used by the Edit morph workflow.       

<img height="600" alt="image" src="https://github.com/user-attachments/assets/c1898eb8-17db-42e8-950a-664ba63201c8" />

-   **Remix**: Upload source audio and restyle it with your own caption and lyrics. The AI uses the original as a structural guide while applying your new style. Adjust Remix Strength to control how closely it follows the original (high = faithful cover, low = loose reinterpretation).    
    -   **What it does:** uses Source Audio as the structural guide for melody, rhythm, chords, arrangement, and timing while applying your new Caption and Lyrics.        
    -   **How to use it:** upload Source Audio, optionally trim it in Source Audio Preview, write the target style in Caption, provide replacement Lyrics if you want changed vocals, then adjust Remix Strength and Remix Melody Retention. Higher Remix Strength follows the source more closely; lower strength gives the model more room to reinterpret.        
    -   **Audio inputs:** Source Audio is the important input here. Reference Audio is only an extra global style cue. If the source is instrumental-only, Remix can follow the instrumental structure but still has to invent the vocal melody and phrasing for new lyrics.        

<img height="600" alt="image" src="https://github.com/user-attachments/assets/a4fb990f-0c83-4709-8609-bd0cb55138fd" />

-   **Repaint**: Upload Source Audio, choose a start/end range, and regenerate only that range. Caption/Lyrics describe the replacement section. Optional Reference Audio can guide style/timbre, but it is not the audio being edited.    
-   Repaint: regenerate one time range of the source    
    -   **What it does:** keeps the Source Audio context and redraws only the selected start/end range. Use it to fix a bad section, replace a lyric phrase, change a solo, or smooth a transition without regenerating the whole song.        
    -   **How to use it:** upload Source Audio, set Repainting Start and End in seconds, then write Caption and Lyrics for the replacement section only. Use Conservative to protect boundaries, Balanced for normal edits, or Aggressive when the selected range should change more freely.       
    -   **Audio inputs:** Source Audio is the audio being edited. Reference Audio can nudge style/timbre for the replacement, but it is not the editable source and will not force exact melody or lyric timing.        

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/13f1262e-d3ee-4687-a3f8-cc69f44d0798" />

-   **Extract**: Isolate a single track (vocals, drums, bass, etc.) from source audio using AI stem separation. Useful for creating instrumentals, acapellas, or isolating parts for remixing. Available on Base only.    
-   **Extract**: isolate one stem from source audio    
    -   **What it does:** separates one selected track from Source Audio, such as vocals, drums, bass, guitar, keyboard, or other supported categories.        
    -   **How to use it:** upload Source Audio, choose Track Name, select the Extract output format if needed, then click Extract Stem. Use the extracted stem for acapellas, instrumentals, remix prep, cleanup, or analysis.        
    -   **Audio inputs:** Extract uses Source Audio only. Caption, Lyrics, Reference Audio, Think, BPM, and key are not creative controls for this mode.        

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/d3602265-132c-4710-ad05-b37fb786b208" />

-   **Lego**: Choose a predefined instrument category such as synth, bass, drums, or guitar. The AI generates that instrument and adds it over the existing source audio; you do not upload external stems. Upload the source track for context, trim it in Source Audio Preview if needed, choose the instrument to add, and describe only that new layer. Available on Base and SFT; Base generally gives the best results.    
-   **Lego**: add one generated track over existing audio    
    -   **What it does:** creates the selected instrument category and layers it over Source Audio. This is for adding a new AI-generated part, not for uploading your own external stem.        
    -   **How to use it:** upload Source Audio, choose Track Name such as vocals, backing\_vocals, drums, bass, guitar, or synth, optionally set the start/end range, then describe only the new layer in Caption. For vocals, provide Lyrics and describe the singer/delivery.        
    -   **Audio inputs:** Source Audio gives musical context for the new layer. Reference Audio can nudge global sound, but it will not act as a guide vocal. If you add vocals to an instrumental, the model must invent the sung melody and phrasing unless the source already contains that vocal structure.        

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/dd979e0f-08b5-4014-806b-f519685557d3" />

-   **Complete**: Fill in selected missing tracks from source audio. Upload a partial arrangement or single stem, trim it in Source Audio Preview if needed, choose the tracks to add, and optionally set Complete Start/End to regenerate only that section while preserving the rest of the source. Available on Base and SFT; Base generally gives the best results.    
-   **Complete**: fill missing tracks in a partial arrangement    
    -   **What it does:** listens to Source Audio and generates the selected missing track classes so the partial idea becomes a fuller arrangement.        
    -   **How to use it:** upload a partial track, single stem, or incomplete mix, choose the track classes to add, optionally set Complete Start and End to limit the generated section, then describe the desired finished arrangement in Caption. Use it for adding accompaniment around vocals, drums/bass under a sketch, or missing instruments in a section.        
    -   **Audio inputs:** Source Audio is the context that the new tracks must fit. Reference Audio can guide overall style, but it does not replace the source and does not force exact melodic or lyric timing.        

<img height="600" alt="image" src="https://github.com/user-attachments/assets/4be71b46-2a58-461d-9c75-6efd3fae552f" />

-   Get latest zip file (4\_3), overwrite previous files and run Windows\_Install\_or\_Update.bat to update

<h3>4 June 2026 V4.4 Update</h3>

- SAM Audio processing bug fixed
- In ACESTEP XL 1.5 Advanced Extract tab, Analyze button was useless now it will show info message to use Track Name and click Extract Stem
- Extract Stem will now show progress and status in Latest Result Status
- Extract Stem limited to ACESTEP XL Base model since it works 100x better with Base than SFT
- Now all advanced tab audio / video input fields will show preview
	- If preview doesn't show immediately, click X and re-select file this fixes Gradio bug
	- Now you can trim audio from Gradio preview as well
- Audio previews visuality improved and trim feature visuality improved significantly for all upload audio fields and previews

<img width="3537" height="482" alt="image" src="https://github.com/user-attachments/assets/00ceb8fd-b07c-4cd4-a571-087dc68c175c" />

- Gradio version upgraded to 6.16.0
- ACESTEP XL 1.5 Advanced Mode Lego and Complete features improved and bugs fixed
- Description of how Lego mode works improved
	- Lego: Choose a predefined instrument category such as synth, bass, drums, or guitar. The AI generates that instrument and adds it over the existing source audio; you do not upload external stems. Upload the source track for context, trim it in Source Audio Preview if needed, choose the instrument to add, and describe only that new layer.
- Get latest zip file (4_3), overwrite previous files and run Windows_Install_or_Update.bat to update 

<h3>3 June 2026 V4.1 Update</h3>

- V4.1 is a massive update so please carefully read all
- For update please get latest v4_1 zip file, extract into install folder, overwrite and run Windows_Install_or_Update.bat file
	- If you get any errors for any reason, delete \ACE-Step_Premium\venv and then run installer bat file
- When you select ACESTEP XL 1.5 SFT or Base model, in advaced tab, now all these options will be enabled and fully work
	- Simple, Custom, Remix, Repaint, Extract, Lego, Complete
   
<img width="1822" height="342" alt="image" src="https://github.com/user-attachments/assets/c81b17aa-9ea2-4d48-bfb1-8c07954032d0" />

- Extract now fully works and you can pick what to extract from Track Name below
	- However I think new SAM Audio model is better still test and compare both

<img height="600" alt="image" src="https://github.com/user-attachments/assets/4000285d-7443-488c-8ba4-6f8e3a20265a" />

- You can also use batch extract feature now if you want to batch process a folder of songs

<img height="600" alt="image" src="https://github.com/user-attachments/assets/0f9b2df7-8383-439b-b763-bc16ba28ac41" />

- Audio Processing tab improved and now we support extremely famous Auto-Editor
- Auto-Editor is amazing library to trim silent - no spoken parts
- I use this to trim out videos and can be very useful to trim vocal extraction
- I use this to also cut silent parts of my tutorials, very useful to pre-process before editing

<img height="600" alt="image" src="https://github.com/user-attachments/assets/33736271-51ce-4320-bb94-b3cd0e904942" />

- New tab SAM Audio Segment implemented
- SAM Audio is state of the art prompt and mask based audio processing / seperation model from Facebook : https://ai.meta.com/research/samaudio/
- It supports any custom text prompt and the below quick select presets

<img height="600" alt="image" src="https://github.com/user-attachments/assets/d1cc0bfe-fbd7-4d79-91f7-c97608438bcb" />

- I have made massive amount of optimizations and programming to implement this model
- BF16 pre-converted safetensors SAM-Audio and SAM-Audio Judge models will be automatically downloaded when you run installer or model downloader bat file
- Normally released models were FP32 .pt models but I converted them to BF16 and safetensors format
- SAM Audio models official pipeline was also first loading into RAM and then moving into GPU thus using extra RAM and slower
	- I made it directly to be loaded into GPU as BF16
- Our implementation supports full sub-process running and auto trim feature - extremely useful to extract vocals for ACESTEP XL 1.5 LoRA vocal training

<img width="3505" height="191" alt="image" src="https://github.com/user-attachments/assets/f9329b1d-9830-4da1-924f-4b07f4b33522" />

- We already have VRAM presets for every GPU out there for SAM Audio model

<img height="600" alt="image" src="https://github.com/user-attachments/assets/59558997-09bb-4f19-b431-95f8083b6fdc" />

- It works with 20 seconds segmentation with 5 seconds overlap
	- 20 seconds segmentation is what model authors recommend and used for training
	- Longer segmentation not improving quality but increases VRAM usage and reduces processing time
- Only missing feature is Multi-diffusion text-only mode since authors didn't publish this but I opened an issue and expecting them to publish hopefully
	- We already have that mode coded by CODEX but I think it is not better due to our inaccurate implementation

<img height="600" alt="image" src="https://github.com/user-attachments/assets/d4a0420b-c5d6-48d0-8716-f01b0e356ac4" />

- We support batch folder processing to pre-process training songs as well or for any reason you want

<img height="600" alt="image" src="https://github.com/user-attachments/assets/22fda4f2-eb06-4f59-ae89-dd98169333d1" />

- Flash Attention were not working on Windows RTX 4000 series GPUs and this issue fixed
	- I have re-compiled Flash Attention 2.8.3 to fully support RTX 3000, 4000 and 5000 series GPUs with extra CUDA Arch a flag for SM120a
	- Linux Flash Attention with all GPUs (to include Cloud server GPUs too) SMs also recompiled and now will be used : 80;86;89;90;100;103;120
	- So no GPU should get any error with Flash Attention anymore
	- More information regarding CUDA archs : https://www.patreon.com/posts/159064759
- ACESTEP XL 1.5 Advanced tab now supports uploading video files as well
	- They will be automatically converted into audio and used
	- If your video upload shows processing forever, click X icon and reupload
		- This is a Gradio bug I am trying to fix, refresh page also fixes
- Audio Processing tab supports both Audio and Video uploading
- SAM Audio Segment supports both Audio and Video uploading
- Video upload previews are now capped to height 400px so they won't take entire web page space and look much better

<h3>31 May 2026 V3.9.1 Update</h3>

- New full audio post-processing tab implemented to our premium app from TrackAICleaner repo
- You can use this tab to both post-process your existing audio files as batch or as single file or automatically post process your generated songs
- When it is enabled to auto post-process generated songs, it will save both original and post-processed songs in the outputs folder
- You can use preview button to generate 60 second preview and compare quickly the effect impact
- Use latest newer zip file, overwrite and run installer to update

<img height="600" alt="image" src="https://github.com/user-attachments/assets/c72f26d1-8287-4d07-827a-bc0a45dbc96c" />

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/4454e2ec-8d68-4540-aeb0-638ff9c963a4" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/1dd63340-b60e-4009-bafc-34fd606ef568" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/f77f9d8b-b8a8-43e3-9579-d15c625dd00d" />


<h3>28 May 2026 V3.9 Update</h3>

- New feature LM Audio Codes added and enabled for all default presets
	- This is supposed to improve quality in all generations without any loss or VRAM increase

<img width="1562" height="262" alt="image" src="https://github.com/user-attachments/assets/39b5c7d4-10a5-40d1-849b-839eb965f0ee" />
   
- Updates made to fix below error that some users reported
	- Sadly I couldn't reproduce it yet to verify
		- Error: Generation produced NaN or Inf latents (shape=[1, 8261, 64], dtype=torch.bfloat16, device=cuda:0, nan=528704, inf=0).
- Same zip file just run installer to update

<h3>26 May 2026 V3.8 Update</h3>

- Version 3.8 is a very major upgrade for training
- In Advanced tab when you click Analyze button now it will auto initialize model and won't throw error
- Custom Preset System save and load issues fixed for some cases
- Now we support DoRA for both training and inference (song generation)
	- DoRA is like LoRA but better quality for training close to full Fine Tuning of the entire model

<img width="1986" height="459" alt="image" src="https://github.com/user-attachments/assets/ba6eb816-2778-4220-a745-80a98a08a9f4" />
   
- Moreover now we have Target MLP feature for training
	- Also applies LoRA/DoRA to decoder MLP layers (gate_proj, up_proj, down_proj). This increases trainable capacity and VRAM use; leave off for the legacy attention-only path.
	- When MLP enabled, more parameters are trained thus it may be a little bit slower and may require more VRAM but it should improve quality I am still in research

<img height="600" alt="image" src="https://github.com/user-attachments/assets/c246fbc7-2076-40dc-958b-44e322afb674" />

- Training Parameters screen significantly improved with lots of new features
- Now we have Save best feature
	- It will save best loss having checkpoint and as new best loss having checkpoint reached, it will overwrite previous best
	- You can set Best smoothing window, Best min delta and Start saving best after epoch as you wish to make it as you wish
- Now we support following Optimizers : adamw, adamw8bit, adafactor
- Now we support following Schedulers : cosine, cosine_restarts, linear, constant, constant_with_warmup
	- I usually prefer constant, I am still in research of best hyper parameters and accurate way of training
- Now we support following Timestep modes : continuous, discrete
- Now we support Adaptive timestep

<img height="600" alt="image" src="https://github.com/user-attachments/assets/d97fbd78-d16a-4c27-abcc-af92f2205a31" />
  
- Now we have an amazing new feature to measure quality of training progress : Validation split %
	- Lets say you have 100 training songs, and you set 5%, so it will set 5 songs as validation and use 95 songs for training
	- Therefore, you can see actual quality improvement or degrade of the model therotically
	- You can see in below screenshot that as the training continued, the validation loss rate got worse and worse even though traidional loss got lower and lower
	- This is because the model got completely overtrained and cooked and memorized and lost its full generalization

<img height="600" alt="image" src="https://github.com/user-attachments/assets/840f6900-a7b8-4404-8eb4-d524f05925c8" />
  
- With the newest features, training VRAM presets are also got updated as below

<img height="600" alt="image" src="https://github.com/user-attachments/assets/5d2716b6-10ee-42dd-b256-cfa58e93480e" />

- Auto LRC feature improved and now uses lesser VRAM moreover generated .lrc and .vtt files are now automatically saved inside generated output folder
- Zip file is same just use Windows_Install_or_Update.bat to update

<h3>25 May 2026 V3.6 Update</h3>

- Brings some massive improvements for training and LoRA usage
- In Generate Song tab now Latest Song section will display status

<img height="600" alt="image" src="https://github.com/user-attachments/assets/e0a9436e-ca22-4777-941c-494f15c9bcb5" />

- Dataset tab completey remade and now when you load your generated dataset.json file, it will  show caption statistics as word n-grams
- You can select word-ngrams and it will list songs containing them
- Then you can select songs to see their full details
- Useful to see your dataset composition

<img height="600" alt="image" src="https://github.com/user-attachments/assets/70f3efd9-a5bb-41e4-830a-7f1e2ca7e73f" />

- Now there is Use Only Custom Trigger checkbox which will make all Caption / Style data to be only as your custom trigger
- I am gonna test this approach to see if works better on training hopefully

<img height="600" alt="image" src="https://github.com/user-attachments/assets/89ba13d0-87fa-403a-8a75-c60f62a918ee" />

- We have a full new Grid Testing tab which lets you to generate songs with selected LoRAs to compare them very easy and quickly 
- So that you can compare your LoRA training checkpoints properly
- You can select multiple LoRAs, any LoRAs you want with filter LoRAs extra feature to make selection easier

<img height="600" alt="image" src="https://github.com/user-attachments/assets/c7d5e90e-ffec-4c76-807e-45eb9678a6c7" />
<img height="600" alt="image" src="https://github.com/user-attachments/assets/692ee3ba-510b-4765-a28e-11d06d3001e4" />

- Grid results will have special naming and save to make them easier to compare e.g. like below

<img height="600" alt="image" src="https://github.com/user-attachments/assets/442f2267-62a7-4593-85c0-d8913f6f6719" />

- Just run installer to update

<h3>24 May 2026 V3.5 Update</h3>

- When auto labeling, even though process finished it was still showing as processing at Raw Lyrics (from .txt file) and this bug fixed
- The training speed display fixed and now it shows training speed accurately after first 20 steps completed like below
- Auto labeling is taking massive time, therefore I have implemented true batch size to this process

<img width="1097" height="334" alt="image" src="https://github.com/user-attachments/assets/316430ca-bcb1-46a1-8f37-95d7e49dc5f5" />

- Cancel auto labelling process and tensor generation process implemented
	- Now you can immediately cancel both processes when running in isolated subprocess
- User / Custom preset system now will save and load every field exists in training tab properly
- Delete preset button now deletes the selected preset and loads the next custom preset
	- If all user custom presets get deleted, it will load default vram preset
- Now your lyric txt files can also contain style / caption 
	- e.g. a.mp3 and a.txt in same folder
	- Format is like below
	- ```# Caption```
		- Your caption
	- ```# Lyrics```
		- Your lyrics
	- I tried to make it as much as possible robust so it should work fairly well see below example
- Custom Trigger Tag was not working properly and this issue fixed
- New feature Debug: save text prompts added so that you can see what is exactly used to generate Preprocessed tensor files which are actually used for training
- Just run installer to update

<img height="600" alt="image" src="https://github.com/user-attachments/assets/4e46ed9c-4bb1-4590-8998-329963a00f3b" />
<img height="600" alt="image" src="https://github.com/user-attachments/assets/e70ad9e3-ccdb-4f4e-a94e-5009b2efd0e0" />
<img height="600" alt="image" src="https://github.com/user-attachments/assets/ac420e7e-d84a-4f8c-85b3-00c6c05f27ee" />


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
	- unlimited (>=24GB), tier6b (24GB safe), tier6a (16-24GB), tier5 (12-16GB), tier4 (8-10GB), tier1 (CPU)
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
| **tier1: CPU / <8GB** | CPU fallback | None (DiT only) | `pt` | LM disabled by default; CPU device selected for this preset |
| **tier4: 8-10GB** | XL turbo with full offload | `acestep-5Hz-lm-0.6B` | `vllm` | Auto also uses this conservative preset for 10-12GB |
| **tier5: 12-16GB** | XL turbo/sft | `acestep-5Hz-lm-1.7B` | `vllm` | INT8 quantization with CPU offload |
| **tier6a: 16-24GB** | XL turbo/sft | `acestep-5Hz-lm-4B` | `vllm` | 4B LM with INT8 quantization and CPU offload |
| **tier6b: 24GB safe** | XL turbo/sft | `acestep-5Hz-lm-4B` | `vllm` | Manual safe preset with CPU offload and no DiT quantization |
| **unlimited: >=24GB** | XL sft (or xl-base for extract/lego/complete) | `acestep-5Hz-lm-4B` | `vllm` | Auto-selected at 24GB or larger; all models stay on GPU |

> **XL (4B) models** (`acestep-v15-xl-*`) offer higher audio quality with ~9GB VRAM for weights (vs ~4.7GB for 2B). They require >=12GB VRAM with offload + quantization, while the no-offload unlimited preset is auto-selected at >=24GB. All LM models are fully compatible with XL.

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
