import os
import chemfp
import tempfile
import random


def csv2fps(csv_file, fp_type='ECFP4', out_type='fpb', nBits=2048, save_name=None,
            separator='\t', run_topN=None, top_sele=True):
    tmp_smi = tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False)
    try:
        with open(csv_file) as fr:
            lines = fr.readlines()
            if isinstance(run_topN, int) and run_topN > 0:
                if top_sele:
                    lines = lines[:run_topN]
                else:
                    if len(lines) > run_topN:
                        lines = random.sample(lines, run_topN)
            for line in lines:
                smi, cid, score = line.strip().split(separator)
                print(f'{smi}\t{cid}\t{score}', file=tmp_smi)
    except:
        with open(csv_file) as fr:
            lines = fr.readlines()
            if isinstance(run_topN, int):
                if top_sele:
                    lines = lines[:run_topN]
                else:
                    if len(lines) > run_topN:
                        lines = random.sample(lines, run_topN)
            for line in lines[1:]:
                items = line.strip().split(separator)
                print(f'{items[0]}\t{items[0]}', file=tmp_smi)

    tmp_smi.close()
    if save_name:
        info = chemfp.ob2fps(tmp_smi.name, save_name, type=fp_type, nBits=nBits)
    else:
        if out_type == 'fps':
            out_file = csv_file.replace('.csv', '.fps')
        elif out_type == 'fpb':
            out_file = csv_file.replace('.csv', '.fpb')
        info = chemfp.ob2fps(tmp_smi.name, out_file, type=fp_type, nBits=nBits)
    os.unlink(tmp_smi.name)  
    return info