# VideoFake: GitHub → Colab

[Apri in Colab](https://colab.research.google.com/github/LucaLorenz/VideoFake/blob/main/VideoFake_colab.ipynb)

Il notebook è un launcher. `colab_runner.py` contiene installazione, controlli e avvio: **viene scaricato dall'ultimo commit di `main` a ogni esecuzione degli step 0, 1, 2, 3 e 5**. La revisione viene risolta prima del download: il codice eseguito corrisponde al commit stampato nell'output.

1. La prima volta riapri il notebook dal link sopra e seleziona GPU T4.
2. Esegui gli step da 0 a 5. Lo step 3 crea Python 3.10 separato, installa le dipendenze e verifica gli import e una piccola inferenza ONNX sulla GPU. La prima installazione richiede diversi minuti.
3. Dopo una modifica nella repo, riesegui lo step indicato: non serve ricopiare il codice o ricaricare la pagina. Se cambiano le dipendenze, riesegui lo step 3.
4. In caso di errore fermati. Incolla nella chat soltanto il blocco `DEBUG_BEGIN … DEBUG_END`. In alternativa salva il notebook con gli output nella stessa repo, tramite **File → Salva una copia in GitHub**, poi scrivi “controlla”. Verifica che gli output siano inclusi nel salvataggio.

Il testo delle celle aperte non si sincronizza in tempo reale. Le celle sono comandi di avvio stabili; gli aggiornamenti riguardano il codice che scaricano. Se cambia il launcher stesso, occorre riaprire il notebook una volta.

I rapporti riportano commit, step, errore e ultime righe del log. I log completi e `debug.json` restano in `/content/videofake-live/`. Non vengono inviati automaticamente a GitHub e non servono credenziali GitHub nel runtime. I file del runtime si perdono quando viene eliminato.

Uno step fallito mostra un riquadro rosso con la causa e termina la cella in errore. Uno step riuscito mostra un riquadro verde. Non proseguire dopo un errore, anche se altri step sono ancora eseguibili.

L'applicazione originale viene scaricata da `neuromodern/VideoFake` al commit `1b4f9f43d42de0b4a2d16ff58075464f676ba8dc`. CUDA/cuDNN di sistema non vengono reinstallati. Python 3.10 e le librerie CUDA 11.8 di PyTorch restano nell'ambiente separato.

Correzioni incluse: compatibilità Python/NumPy, PyTorch/Torchvision fissati per GFPGAN, import GUI rinviato alla sola modalità desktop, opzione `--output-video-encoder`, encoder VP9 e significato di `KeepAudio`. I percorsi con spazi vengono passati senza shell.

Controlli locali: `python3 -m unittest discover -s tests -v`. Questi controlli non sostituiscono la verifica GPU dello step 3 su Colab.

## Runpod

Nel Pod, clona la repo in `/workspace/VideoFake-tools`. Lancia `python runpod.py setup` una volta per installare e verificare l'ambiente. Apri `Runpod_VideoFake.ipynb` in JupyterLab, esegui 0, poi 4 e 5. Carica gli input in `/workspace/input`; i risultati vanno in `/workspace/output`.

Da terminale: `python runpod.py run --source /workspace/input/face.jpg --target /workspace/input/video.mp4 --output /workspace/output/out.mp4`. Aggiungi `--no-enhancer` per una prova con il solo swap.

L'elaborazione mostra il progresso in Jupyter o nel terminale; il log completo resta sul Pod. Per aggiornare il checkout: `git pull --ff-only`. Dipendenze, interprete gestito e modelli sono in `/workspace/videofake-live`.
