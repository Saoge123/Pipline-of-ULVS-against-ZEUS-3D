import os
import glob
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import xlsxwriter
from io import BytesIO

    

class MakeExcel(object):
    def __init__(self, score_cutoff, einternal_cutoff, n_head, sort_key, occur_times):
        self.score_cutoff = score_cutoff
        self.einternal_cutoff = einternal_cutoff
        self.n_head = n_head
        self.sort_key = sort_key
        self.occur_times = occur_times

    def parse_docking_results(self, docking_score_files):
        D_hits = {}
        D = {}
        for ix,i in enumerate(sorted(docking_score_files)):
            df = pd.read_csv(i)

            # Step 1: filter docking results by docking status
            df = df[df["docking_status"] == "Done"]

            # Step 2: filter docking results by GlideScore cutoff
            df = df[df[self.sort_key] < self.score_cutoff]

            #print(df.shape)
            # Step 3: filter docking results by internal energy cutoff
            df = df[df["r_i_glide_einternal"] < self.einternal_cutoff]
            
            # Step 4: select the pose with the lowest r_i_glide_emodel for each title
            df_best = df.loc[df.groupby("title")["r_i_glide_emodel"].idxmin()]

            # Step 5: sort the poses by GlideScore and select the top candidates
            df_best = df_best.sort_values(by=self.sort_key)
            
            df_top = df_best.head(self.n_head)

            D_hits[i], smi_dict = [], {}
            for idx, (_,line) in enumerate(df_top.iterrows()):
                smi_dict[line.title] = line.to_dict()
            
            D[i] = smi_dict
            if len(D_hits[i]) == 0:
                del D_hits[i]
        return D

    def get_intersection(self, filtered_results):
        mol_cross = {}
        smi_union = {}
        for i in filtered_results:
            for j in filtered_results[i]:
                if j in smi_union:
                    if filtered_results[i][j][self.sort_key] < smi_union[j][self.sort_key]:
                        smi_union[j] = filtered_results[i][j]
                else:
                    smi_union[j] = filtered_results[i][j]
                if j not in mol_cross:
                    mol_cross[j] = 1
                else:
                    mol_cross[j] += 1

        mols = {}
        for i in mol_cross:
            if smi_union[i][self.sort_key] <= -12:
                mols[i] = smi_union[i]
            elif mol_cross[i] >= self.occur_times:
                mols[i] = smi_union[i]
        #print(len(mols))
        mols = pd.DataFrame(mols).T
        mols = mols.sort_values(by=self.sort_key)
        return mols
    
    def get_excel(self, mols, save_name):
        # 1. limit the number of molecules to be written into Excel
        mols = mols.head(self.n_head)

        # 2. make new Excel sheet 'Structures' with xlsxwriter
        book  = xlsxwriter.Workbook(save_name)
        sheet = book.add_worksheet('Structures')

        # 3. define header format
        header_fmt = book.add_format({'bold': True, 'align': 'center'})

        # 4. write table header
        headers = ['Structure','SMILES', 'Name', 'MW', self.sort_key, 's_i_glide_gridfile']   # 按自己需要改
        for col, title in enumerate(headers):
            sheet.write(0, col, title, header_fmt)

        # 5. set row height and column width (in pixels)
        sheet.set_column('A:A', 30)      # set column width for Structure column
        sheet.set_row(0, 25)             # set row height for header row

        # 6. traverse the molecules in the DataFrame and write them into Excel
        for row, (_, line) in enumerate(mols.iterrows(), start=1):#enumerate(screen_results.iterrows(), start=1):
            smi = line.SMILES
            mol_id = line.title
            mol = Chem.MolFromSmiles(smi)
            if mol is None:                     # skip if SMILES parsing failed
                sheet.write('Invalid SMILES', 1, smi)
                sheet.write(row, 1, smi)
                sheet.write(row, 2, mol_id)
                sheet.write(row, 3, 'none')
                sheet.write(row, 4, line[self.sort_key])
                sheet.write(row, 5, line['s_i_glide_gridfile'])
                continue
            
            mw = Descriptors.MolWt(mol)

            # 6-1 generate PIL image
            pil_img = Draw.MolToImage(mol, size=(200, 200))

            # 6-2 save PIL image to memory (in PNG format)
            buf = BytesIO()
            pil_img.save(buf, format='PNG')
            buf.seek(0)

            # 6-3 write SMILES, Name, MW, GlideScore, and gridfile to Excel
            sheet.write(row, 1, smi)
            sheet.write(row, 2, mol_id)   # write Name to Excel
            sheet.write(row, 3, mw)
            sheet.write(row, 4, line[self.sort_key])
            sheet.write(row, 5, line['s_i_glide_gridfile'])
            
            # 6-4 insert PIL image into cell
            sheet.insert_image(row, 0, 'dummy.png',
                            {'image_data': buf,      # use memory data
                                'x_scale': 1,
                                'y_scale': 1,
                                'object_position': 1})   # image moves with cell resizing
            sheet.set_row(row, 150)                     # set row height for image row

        # 7. save Excel file
        book.close()
        print(f'Excel file {save_name} has been generated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Excel file from ZEUS-3D results')
    parser.add_argument('--docking_path', type=str, required=True, help='Input CSV file')
    parser.add_argument('--save_name', type=str, default=None, help='Output Excel file')
    parser.add_argument('--n_head', type=int, default=10000, help='Number of top candidates to include')
    parser.add_argument('--score_cutoff', type=float, default=-7.0, help='GlideScore cutoff')
    parser.add_argument('--einternal_cutoff', type=float, default=10.0, help='Internal energy cutoff')
    parser.add_argument('--occur_times', type=int, default=3, help='Minimum number of times a molecule must be hit')
    parser.add_argument('--sort_key', type=str, default='r_i_docking_score', choices=['r_i_docking_score', 'r_i_glide_gscore'], help='Sort key for GlideScore')
    """
    Example usage:
    python 6_make_excel.py --docking_path /path/to/ulvs/FLT3 --save_name FLT3_results.xlsx --n_head 10000 --score_cutoff -7.0 --einternal_cutoff 10.0 --occur_times 3 --sort_key r_i_glide_score
    """


    args = parser.parse_args()
    docking_path = args.docking_path
    docking_files = glob.glob(f'{args.docking_path}/5_docking_pose/*-docking_config_SP.csv')
    excel_processer = MakeExcel(args.score_cutoff, args.einternal_cutoff, args.n_head, args.sort_key, args.occur_times)
    results = excel_processer.parse_docking_results(docking_files)
    mols = excel_processer.get_intersection(results)
    if args.save_name is None:
        save_name = f'{os.path.split(docking_path)[0]}/results_from_ZEUS-3D.xlsx'
    else:
        save_name = args.save_name
    excel_processer.get_excel(mols, save_name)

    
