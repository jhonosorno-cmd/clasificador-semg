"""Compara argmax contra la regla de umbral en validación y prueba, pipeline causal.

Si el umbral gana en validación pero pierde en prueba, el barrido se sobreajustó.
Si pierde en las dos, la regla dejó de aportar bajo preprocesamiento causal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))

import src.data_loader as dl
from reentrenar_causal import (COMPONENTES, SUJETO, TEST_REPS, VAL_REPS, CKPT,
                               NinaProExtendedFeatures, preprocesar_causal)

dl.NinaProDB3Dataset._preprocess_emg = preprocesar_causal

modelos = []
for alias, (Clase, dim, _, _) in COMPONENTES.items():
    m = Clase(manual_features_dim=dim)
    m.load_state_dict(torch.load(CKPT / f"{alias}_causal.pth", weights_only=False)["model_state_dict"])
    m.eval(); modelos.append((m, dim))

def probs_de(reps):
    d48 = dl.NinaProDB3Dataset(subjects=[SUJETO], reps_to_use=reps, mode="test",
                               augment=False, target_subject=SUJETO)
    d132 = NinaProExtendedFeatures(subjects=[SUJETO], reps_to_use=reps, mode="test",
                                   augment=False, target_subject=SUJETO)
    s = torch.from_numpy(np.asarray(d48.samples, dtype=np.float32))
    f48 = torch.from_numpy(np.stack([d48._extract_manual_features(x[0, :, :12]) for x in d48.samples])).float()
    f132 = torch.from_numpy(np.stack([d132._extract_manual_features(x[0, :, :12]) for x in d132.samples])).float()
    acum = None
    with torch.no_grad():
        for m, dim in modelos:
            p = torch.softmax(m(s, f132 if dim == 132 else f48), dim=1)
            acum = p if acum is None else acum + p
    return (acum / len(modelos)).numpy(), np.asarray(d48.labels)

def umbral(p, t):
    return np.where(p[:, 0] > t, 0, p[:, 1:].argmax(1) + 1)

vp, vy = probs_de(VAL_REPS)
tp, ty = probs_de(TEST_REPS)

rejilla = np.arange(0.10, 1.00, 0.02)
accs_val = np.array([(umbral(vp, t) == vy).mean() for t in rejilla])
mejor_t = float(rejilla[int(accs_val.argmax())])

print(f"\n{'':28}{'VALIDACIÓN':>22}{'PRUEBA':>22}")
print(f"{'regla':28}{'exact.':>11}{'balanc.':>11}{'exact.':>11}{'balanc.':>11}")
filas = [("argmax", vp.argmax(1), tp.argmax(1)),
         (f"umbral θ={mejor_t:.2f} (óptimo val)", umbral(vp, mejor_t), umbral(tp, mejor_t)),
         ("umbral θ=0,36 (documento)", umbral(vp, 0.36), umbral(tp, 0.36))]
for nombre, pv, pt in filas:
    print(f"  {nombre:26}{(pv==vy).mean()*100:10.2f}%{balanced_accuracy_score(vy,pv)*100:10.2f}%"
          f"{(pt==ty).mean()*100:10.2f}%{balanced_accuracy_score(ty,pt)*100:10.2f}%")

mejor_arg_val = (vp.argmax(1) == vy).mean() * 100
print(f"\n  ¿el umbral óptimo supera al argmax en validación? "
      f"{'sí' if accs_val.max()*100 > mejor_arg_val else 'NO'} "
      f"({accs_val.max()*100:.2f} vs {mejor_arg_val:.2f})")
