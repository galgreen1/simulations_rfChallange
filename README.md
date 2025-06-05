# RF Challenge Simulations

פרויקט סימולציות RF הכולל:
- `simulations.py`    : ריצת סימולציה שמחשבת BER ושומרת `outputs1.txt`.
- `replot.py`         : קורא מ־`outputs1.txt` ויוצר גרף `ber_replot.png`.
- `docker/Dockerfile` : Dockerfile לבניית התמונה עם כל התלויות.
- `slurm/run.slurm`   : סקריפט SLURM שמשגר את הקונטיינר ומריץ סימולציה + replot.

## מבנה התיקיות

