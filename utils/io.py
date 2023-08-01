import scipy.io as scio
import os
from pathlib import Path

def save_reconstructions(recons, save_dir):
    
    for fname, recon in recons.items():
        file_parts = fname.split("/")
        out_dir = Path(os.path.join(save_dir, file_parts[-3], file_parts[-2]))
        out_dir.mkdir(exist_ok=True, parents=True)
        path = (out_dir / file_parts[-1]).resolve()
        save_dict = {'img4ranking': recon.transpose(1,2,0,3)}
        scio.savemat(file_name=str(path),
                            mdict=save_dict)