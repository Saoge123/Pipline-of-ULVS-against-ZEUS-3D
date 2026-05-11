import random
from tqdm.auto import tqdm
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from ulvs_pipline.ulvs_utils.utils.sascorer import calculateScore


class MolFilter(object):
    def __init__(self, mw_max=700, mw_min=200, logp=5, hbd=5, hba=10, qed=0.0, score=0.0, sa=10, substruct_list=[]):
        self.mw_max = mw_max
        self.mw_min = mw_min
        self.logp = logp
        self.hbd = hbd
        self.hba = hba
        self.qed = qed
        self.score = score
        self.sa = sa
        self.substruct_list = [Chem.MolFromSmarts(m) for m in [Chem.MolToSmarts(Chem.MolFromSmiles(i)) for i in substruct_list]]

    def lipinski_filter(self, mol):
        """检查分子是否符合 Lipinski 五规则"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        violations = 0
        if mw > self.mw_max or self.mw_min <= 200: violations += 1
        if logp > self.logp: violations += 1
        if hbd > self.hbd: violations += 1
        if hba > self.hba: violations += 1
        return violations <= 1

    def substructure_match(self, mol):
        """检查分子是否符合子结构列表"""
        return any(mol.HasSubstructMatch(substruct) for substruct in self.substruct_list)

    def mol_filter(self, line):
        try:
            if float(line[-1]) < self.score:
                return False
            smi = line[0]
            m = Chem.MolFromSmiles(smi)
            if len(self.substruct_list) > 0:
                has_substruct = self.substructure_match(m)
            else:
                has_substruct = False
            is_qed = QED.qed(m) > self.qed
            is_lipinski = self.lipinski_filter(m)
            sa_check = calculateScore(m) < self.sa
            if is_qed and is_lipinski and not has_substruct and sa_check:
                return [line[0], line[1], float(line[-1])]
            else:
                return False
        except:
            return False
    
    def __call__(self, csv_file, top_sele=True, run_topN=None, n_process=8, use_filter=True):
        if isinstance(csv_file, str):
            with open(csv_file) as fr:
                if isinstance(run_topN, int):
                    if top_sele:
                        lines = [i.strip().split('\t') for i in fr][:run_topN]
                    else:
                        Lines = fr.readlines()
                        if len(Lines) <= run_topN:
                            lines = [i.strip().split('\t') for i in Lines]
                        else:
                            lines = [i.strip().split('\t') for i in random.sample(Lines, run_topN)]
                else:
                    lines = [i.strip().split('\t') for i in fr]
        else:
            if isinstance(run_topN, int):
                if top_sele:
                    lines = csv_file[:run_topN]
                else:
                    if len(csv_file) <= run_topN:
                        lines = csv_file
                    else:
                        lines = random.sample(csv_file, run_topN)
            else:
                lines = csv_file
        if use_filter:
            with Pool(n_process) as pool:
                out = pool.map(self.mol_filter, tqdm(lines))
        else:
            out = lines
        
        filtered_lines = sorted([i for i in out if i], key=lambda x: x[-1], reverse=True)
        return filtered_lines