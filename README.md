# VideoFake: nuovo Pod → un comando → notebook pronto

Codice, installer e notebook sono in questa repo. Non occorre copiare il vecchio Pod, costruire immagini Docker o acquistare storage persistente. I modelli vengono scaricati automaticamente dai rispettivi hosting.

## Nuovo Pod

1. Crea un Pod NVIDIA con template ufficiale **Runpod PyTorch**, **JupyterLab** attivo e **50 GB di container disk**. Non serve un volume persistente. Stack compatibile con A40, RTX A5000/A6000, RTX 3090/4090. RTX 50/Blackwell richiedono un ambiente più recente e vengono bloccate prima di installare.
2. Apri il terminale del Pod ed esegui:

```bash
curl -fsSL https://raw.githubusercontent.com/LucaLorenz/VideoFake/main/bootstrap_runpod.sh -o /tmp/videofake-install.sh && bash /tmp/videofake-install.sh
```

3. Attendi **PRONTO**. Lo script scarica il progetto, crea Python 3.10 separato, installa le dipendenze, verifica CUDA e scarica i modelli. **Non avvia test video.** Se fallisce si ferma con errore visibile.
4. In JupyterLab apri **`/workspace/VideoFake-tools/Runpod_VideoFake.ipynb`**.
5. Carica foto e video in **`/workspace/input`**. Esegui **0**, configura i percorsi nello **step 4**, poi esegui **5**. Salta 1–3: installazione già completata.
6. Scarica i risultati da **`/workspace/output`**. Poi puoi eliminare il Pod dopo aver salvato ciò che ti interessa.

Su qualsiasi nuovo Pod compatibile ripeti gli stessi passaggi. Nessun dato del vecchio Pod è necessario.

## Installazione

- Codice e notebook: `/workspace/VideoFake-tools`.
- Python, librerie e modelli: `/opt/videofake`, sul disco locale del container. Evita l'installazione lenta di migliaia di file su storage di rete.
- Configurazione automatica: `.runtime.json`, letta dal notebook e da `runpod.py`.
- Modelli: circa 1,2 GB di download; vengono controllati e riutilizzati se già presenti.
- Input e risultati: `/workspace/input` e `/workspace/output`.

**Senza storage persistente, arresto/eliminazione può cancellare i file. Scarica i risultati prima.** L'ambiente è ricreabile dalla repo.

L'obiettivo è un avvio di pochi minuti, ma il tempo non è garantito: dipende da download, CPU, disco e disponibilità GPU. Non occorre tenere acceso il vecchio Pod.

L'installer è rieseguibile: se l'ambiente è completo, verifica soltanto dipendenze, GPU e modelli. Non sovrascrive notebook modificati e non aggiorna automaticamente un checkout esistente.

## Uso

Gli step del notebook scaricano il runner dall'ultimo commit di `main` a ogni Esegui. Se cambiano le dipendenze, chiedono di rieseguire lo step 3.

I nuovi notebook partono con **solo swap e un solo volto**. Per aggiungere l'enhancer usa `Processor = "face_swapper face_enhancer"`. Progresso, frame e tempo stimato sono visibili. Dimensioni e proporzioni sono ricavate dinamicamente dall'input; i metadati FPS anomali non vengono usati come frame rate.

Da terminale, nella cartella progetto:

```bash
python3 runpod.py run --source /workspace/input/face.jpg --target /workspace/input/input_video.mp4 --output /workspace/output/out.mp4
```

Aggiungi `--enhancer` o `--many-faces` solo se desiderato. Per aggiornare un checkout senza modifiche locali: `git pull --ff-only`.

## Errori e verifiche

Gli errori mostrano un riquadro rosso e fanno fallire la cella. Condividi soltanto `DEBUG_BEGIN … DEBUG_END`. I log completi restano nel runtime. Nessuna chiave GitHub richiesta e nessun invio automatico dei rapporti.

Controlli software: `python3 -m unittest discover -s tests -v`. I test video restano manuali.

## Colab e ambienti precedenti

[Apri Colab](https://colab.research.google.com/github/LucaLorenz/VideoFake/blob/main/VideoFake_colab.ipynb). Senza `.runtime.json`, i Pod precedenti usano il profilo originale. L'installer non sposta o cancella vecchi ambienti.

Basato su [VideoFake di NeuroModern](https://github.com/neuromodern/VideoFake), commit applicativo `1b4f9f43d42de0b4a2d16ff58075464f676ba8dc`.
