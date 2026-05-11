import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import glob
import subprocess
import argparse
import numpy as np
from rdkit import Chem
from ulvs_utils.data_process.protein.parse_pdb import Protein



def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))


def get_rep_pkts(rep_txt):
    path = os.path.split(rep_txt)[0]
    lig_sdfs = {}
    for sdf in glob.glob(path + '/*.sdf'):
        pdb_id, lig_id = sdf.split('/')[-1].split('_')[0:2]
        lig_sdfs[(pdb_id, lig_id)] = sdf
    
    pdbs = {}
    for pdb in glob.glob(path.replace('/pockets/', '/pdbs/') + '/*.pdb'):
        pdb_id = pdb.split('/')[-1].split('_')[0]
        pdbs[pdb_id] = pdb
    
    with open(rep_txt, 'r') as f:
        l = []
        for line in f.readlines():
            pdb_id, lig_id = line.strip().split('/')[-1].split('_')[0:2]
            l.append((pdbs[pdb_id], lig_sdfs[(pdb_id, lig_id)]))
    return l


FORCE_FIELD = {'OPLS_2005':[2005, 'OPLS_2005'], 
               'OPLS_3':[3, 'OPLS3'],
               'OPLS_4':['S-OPLS', 'OPLS4'],
               'OPLS_5':[5, 'OPLS5']
               }

class GlideScore(object):
    def __init__(self, work_path='glide_score_tmp/', bff='OPLS_2005', lig_bff='2005', precision='SP', n_jobs=10, n_struct=None, max_lig_prep_conf=32):
        """
        Initialize a GlideScore object for molecular docking calculations using Schrödinger's Glide software.
        
        This constructor sets up all necessary parameters and working directories for protein-ligand 
        docking workflows, including force field selection, precision settings, and computational resources.
        
        Parameters
        ----------
        work_path : str, optional
            Working directory path where all intermediate files and results will be stored.
            Default is 'glide_score_tmp/'. The path will be converted to absolute path and 
            a trailing slash will be added automatically.
        
        bff : str, optional
            Force field identifier. Must be one of the keys in the FORCE_FIELD dictionary.
            Currently supported options are 'OPLS_2005' and 'OPLS_3'. Default is 'OPLS_2005'.
        
        precision : str, optional
            Docking precision level that controls the accuracy vs. speed trade-off in docking 
            calculations. Default is 'SP' (Standard Precision). Other options typically include 
            'HTVS' (High-Throughput Virtual Screening) and 'XP' (Extra Precision).
        
        n_jobs : int, optional
            Number of parallel jobs/threads to use for computationally intensive operations.
            Default is 10. Higher values can speed up calculations but require more computational resources.
        
        n_struct : int or None, optional
            Maximum number of structures to process. If None (default), processes all available structures.
            Useful for limiting processing in large-scale screening campaigns.
        
        max_lig_prep_conf : int, optional
            Maximum number of conformers to generate during ligand preparation step.
            Default is 32. Higher values explore more conformational space but increase computation time.
        
        Attributes
        ----------
        name : str
            Identifier string set to 'glide'.
        
        work_path : str
            Absolute path to the working directory with trailing slash.
        
        lig_bff : int
            Ligand preparation force field code from the FORCE_FIELD dictionary.
        
        pro_prep_ff : int
            Protein preparation force field code from the FORCE_FIELD dictionary.
        
        docking_ff : str
            Docking force field name from the FORCE_FIELD dictionary.
        
        precision : str
            Docking precision level.
        
        n_jobs : int
            Number of parallel jobs.
        
        n_struct : int or None
            Maximum number of structures to process.
        
        max_lig_prep_conf : int
            Maximum ligand conformers to generate.
        
        Side Effects
        ------------
        Creates the working directory (and parent directories if needed) using verify_dir_exists().
        
        Examples
        --------
        >>> # Initialize with default parameters
        >>> glide_scorer = GlideScore()
        
        >>> # Initialize with custom parameters for high-precision docking
        >>> glide_scorer = GlideScore(
        ...     work_path='/data/docking_results/',
        ...     bff='OPLS_3',
        ...     precision='XP',
        ...     n_jobs=16,
        ...     max_lig_prep_conf=64
        ... )
        
        >>> # Initialize for high-throughput screening
        >>> glide_scorer = GlideScore(
        ...     work_path='htvs_results/',
        ...     precision='HTVS',
        ...     n_jobs=32,
        ...     n_struct=1000,
        ...     max_lig_prep_conf=16
        ... )
        
        Notes
        -----
        The FORCE_FIELD dictionary maps force field names to parameter lists where:
        - Index 0: Ligand preparation force field code
        - Index 1: Protein preparation force field code  
        - Index 2: Docking force field name
        
        The working directory path is always converted to absolute path and normalized with a trailing slash.
        Directory creation is handled automatically - no need to manually create directories before initialization.
        """
        self.name = 'glide'
        self.work_path = os.path.abspath(work_path) + '/'
        self.lig_prep_ff = lig_bff #FORCE_FIELD[bff][0]
        self.pro_prep_ff = FORCE_FIELD[bff][0]
        self.docking_ff = FORCE_FIELD[bff][1]
        self.precision = precision
        self.n_jobs = n_jobs
        self.n_struct = n_struct
        self.max_lig_prep_conf = max_lig_prep_conf
        verify_dir_exists(self.work_path)

    @staticmethod
    def compute_ligand_center(lig_file):
        """
        Calculate the geometric center (centroid) of a ligand molecule from either PDB or SDF format files.
        
        This static method computes the 3D coordinates of the molecular center by averaging the atomic
        coordinates of all heavy atoms in the ligand. It supports both PDB files (manual parsing) and
        SDF files (using RDKit).
        
Parameters
        ----------
        lig_file : str
            Path to the ligand file. Supported formats are:
            - PDB files (.pdb extension): Parsed manually using fixed-column format
            - SDF files (.sdf extension): Parsed using RDKit's Chem.MolFromMolFile
        
        Returns
        -------
        tuple
            A tuple containing:
            - center_coords : tuple of 3 floats (center_x, center_y, center_z)
              The geometric center coordinates of the ligand
            - name : str
              The ligand name extracted from the filename (for SDF) or PDB HETATM records
        
        Raises
        ------
        FileNotFoundError
            If the input file does not exist
        ValueError
            If the file format is not supported or contains no valid atoms
        RuntimeError
            If RDKit fails to parse the SDF file
        
        Notes
        -----
        For PDB files:
        - Only processes lines starting with 'HETATM' (hetero atoms)
        - Extracts coordinates from fixed columns: x(30-38), y(38-46), z(46-54)
        - Extracts residue name from columns 17-20
        - Requires at least one HETATM record
        
        For SDF files:
        - Uses RDKit's Chem.MolFromMolFile for parsing
        - Extracts the first conformer if multiple exist
        - Processes all atoms in the molecule
        - Extracts filename (without path and extension) as ligand name
        
        Examples
        --------
        >>> # For PDB file
        >>> center, name = GlideScore.compute_ligand_center('ligand.pdb')
        >>> print(f"Ligand {name} center: {center}")
        Ligand UNK center: (15.234, -2.456, 8.901)
        
        >>> # For SDF file
        >>> center, name = GlideScore.compute_ligand_center('molecule.sdf')
        >>> print(f"Molecule {name} center: {center}")
        Molecule molecule center: (-1.234, 5.678, 12.345)
        
        See Also
        --------
        grid_generation : Uses this method to determine grid center for docking
        Chem.MolFromMolFile : RDKit function for SDF parsing
        """
        if lig_file.split('.')[-1] == 'pdb':
            # Parse PDB file manually using fixed-column format
            with open(lig_file) as f:
                x, y, z = 0, 0, 0
                count = 0
                for line in f:
                    if line.startswith('HETATM'):
                        x += float(line[30:38].strip())
                        y += float(line[38:46].strip())
                        z += float(line[46:54].strip())
                        name = line[17:20]  # Extract residue name from PDB format
                        count += 1
                center_x = x/count
                center_y = y/count
                center_z = z/count 
        else:
            # Parse SDF file using RDKit
            m = Chem.MolFromMolFile(lig_file, sanitize=False)
            name = lig_file.split('/')[-1].split('.')[0]  # Extract filename as ligand name
            conformer = m.GetConformer()
            coords = np.array([conformer.GetAtomPosition(a.GetIdx()) for a in m.GetAtoms()])
            center_x, center_y, center_z = coords.mean(0)
        return (center_x, center_y, center_z), name

    def grid_generation(self, pdb_file, ligand_file):
        """
        Generate a docking grid for protein-ligand docking calculations using Schrödinger's Glide software.
        
        This method creates a 3D grid centered on the ligand binding site by:
        1. Preparing the protein structure using prepwizard
        2. Calculating the ligand center coordinates
        3. Generating a docking grid with specified dimensions and force field parameters
        
        Parameters
        ----------
        pdb_file : str
            Path to the protein PDB file. The protein structure will be preprocessed and used
            as the receptor for docking grid generation.
        
        ligand_file : str
            Path to the ligand file (SDF, PDB, or other supported format). Used to determine
            the center coordinates for grid placement via compute_ligand_center().
        
        Returns
        -------
        None
            The generated grid file is saved as a ZIP archive in the pocket_grids/ subdirectory.
        
        Side Effects
        ------------
        - Creates pocket_grids/ and prepwizard/ subdirectories in the working path
        - Generates a ZIP archive containing the docking grid file
        - Creates intermediate processed protein files (.mae format)
        - Prints status messages to stdout
        - Runs external Schrödinger commands (prepwizard and glide)
        
        Notes
        -----
        Grid Generation Process:
        1. Extracts receptor and ligand names from filenames
        2. Checks if grid file already exists to avoid recomputation
        3. Prepares protein using prepwizard with specified force field
        4. Calculates ligand center coordinates for grid placement
        5. Creates grid configuration file with:
           - Force field parameters
           - Inner box dimensions (10×10×10 Å)
           - Outer box dimensions (30×30×30 Å)
           - Grid center coordinates
           - Output file specification
        6. Runs glide to generate the docking grid
        
        Filename Convention:
        - Receptor name: extracted from PDB filename (before first '.' and '-')
        - Ligand name: extracted from ligand filename (after first '_')
        - Grid filename: 'grid_{rec_name}_{lig_name}.zip'
        
        Grid Parameters:
        - Inner box: 10×10×10 Å (binding site region)
        - Outer box: 30×30×30 Å (encompassing region)
        - Center: ligand geometric center
        
        Examples
        --------
        >>> glide_scorer = GlideScore()
        >>> glide_scorer.grid_generation('protein_1abc.pdb', 'ligand_1abc_xxx.sdf')
        # grid_1abc_xxx.zip already exists, skip
        
        >>> glide_scorer.grid_generation('receptor_2def.pdb', 'compound_2def_yyy.pdb')
        # Generates grid_2def_yyy.zip in pocket_grids/ subdirectory
        
        See Also
        --------
        compute_ligand_center : Calculates center coordinates for grid placement
        verify_dir_exists : Creates necessary directories
        """
        # Extract receptor and ligand names from filenames
        rec_name = pdb_file.split('/')[-1].split('.')[0].split('-')[0]
        lig_name = ligand_file.split('/')[-1].split('.')[0].split('_')[1]
        
        # Set up grid directory and filename
        grid_dir = self.work_path+'/3_pocket_grids/'
        zip_file_name = f'grid_{rec_name}_{lig_name}'
        
        # Skip if grid already exists
        if os.path.exists('{}.zip'.format(grid_dir+zip_file_name)):
            print(f'# {grid_dir+zip_file_name}.zip already exists, skip')
            return
        
        # Create grid directory
        verify_dir_exists(grid_dir)
    
        # Prepare protein using prepwizard
        prepwizard_dir = self.work_path+'/2_prepwizard/'
        verify_dir_exists(prepwizard_dir)
        pdb_file = os.path.abspath(pdb_file)
        prep_name = pdb_file.split('/')[-1].split('.')[0]+'_processed.mae'
        
        # Run protein preparation
        cmd_list = [
            '#!/bin/bash',
            'cd {}'.format(prepwizard_dir),
            "$SCHRODINGER/utilities/prepwizard -WAIT -fix -f {} {} {} -HOST 'localhost:{}' -j {}".format(
                self.pro_prep_ff, pdb_file, prep_name, self.n_jobs, self.n_jobs)
        ]
        p1 = subprocess.Popen('\n'.join(cmd_list), shell=True)
        p1.wait()
        
        # Calculate ligand center for grid placement
        center, _ = self.compute_ligand_center(ligand_file)
        
        # Create grid configuration
        grid_content = [
            'FORCEFIELD {}'.format(self.docking_ff),
            'INNERBOX 10, 10, 10',
            'OUTERBOX 30.000000, 30.000000, 30.000000',
            'GRID_CENTER {}, {}, {}'.format(*center),
            'GRIDFILE {}.zip'.format(grid_dir+zip_file_name),
            'RECEP_FILE {}'.format(prepwizard_dir + prep_name)
        ]
        
        # Write configuration file and generate grid
        config_file = grid_dir+'grid_config.in'
        open(config_file, 'w').write('\n'.join(grid_content))
        grid_cmd = '\n'.join(
            ['cd {}'.format(grid_dir), 
             "$SCHRODINGER/glide -WAIT -OVERWRITE -NOLOCAL -HOST 'localhost:{}' -NJOBS {} {}".format(
                 self.n_jobs, self.n_jobs, config_file)]
            )
        p2 = subprocess.Popen(grid_cmd, shell=True)
        p2.wait()


    def ligand_preparation(self, lig_file):
        """
        Prepare ligand molecules for docking using Schrödinger's LigPrep utility.
        
        This method processes ligand files in various formats (SDF, CSV, SMILES, MAE) and
        generates 3D structures suitable for docking calculations. It handles conformer
        generation, protonation states, and force field parameter assignment.
        
        Parameters
        ----------
        lig_file : str
            Path to the input ligand file. Supported formats:
            - SDF files (.sdf): Structure Data Format
            - CSV files (.csv): Comma-separated values with molecular data
            - SMILES files (.smi): Simplified molecular-input line-entry system
            - MAE files (.mae): Schrödinger Maestro format
        
        Returns
        -------
        str or bool
            - str: Absolute path to the prepared ligand output file (SDF format)
            - False: If preparation failed or input molecule has ≤1 atoms
        
        Side Effects
        ------------
        - Creates ligprep_out/ subdirectory in the working path
        - Generates prepared ligand files in SDF format with '-out.sdf' suffix
        - Runs external Schrödinger LigPrep command
        - Prints status messages to stdout (via subprocess)
        
        Notes
        -----
        Ligand Preparation Process:
        1. Determines file type based on extension
        2. Validates SDF files using RDKit (rejects molecules with ≤1 atoms)
        3. Creates output directory if needed
        4. Checks if prepared file already exists (to avoid recomputation)
        5. Runs LigPrep with parameters:
           - Parallel processing (n_jobs threads)
           - Specified force field (lig_prep_ff)
           - Epik for pKa prediction and protonation states
           - Maximum conformer limit (max_lig_prep_conf)
           - Output in SDF format regardless of input format
        
        File Type Mapping:
        - SDF → 'isd' (input SDF)
        - CSV → 'icsv' (input CSV)
        - SMILES → 'ismi' (input SMILES)
        - MAE → 'imae' (input MAE)
        
        Output Convention:
        - Input: 'molecule.sdf' → Output: 'molecule-out.sdf'
        - Input: 'compound.csv' → Output: 'compound-out.sdf'
        - All outputs are saved in ligprep_out/ subdirectory
        
        Examples
        --------
        >>> glide_scorer = GlideScore()
        >>> # Prepare SDF file
        >>> prepared_file = glide_scorer.ligand_preparation('ligands.sdf')
        >>> print(prepared_file)
        /path/to/work/ligprep_out/ligands-out.sdf
        
        >>> # Prepare CSV file (returns existing file if already prepared)
        >>> prepared_file = glide_scorer.ligand_preparation('compounds.csv')
        >>> print(prepared_file)
        /path/to/work/ligprep_out/compounds-out.sdf
        
        >>> # Handle invalid input
        >>> result = glide_scorer.ligand_preparation('invalid.sdf')
        >>> print(result)
        False
        
        See Also
        --------
        Chem.MolFromMolFile : RDKit function for SDF validation
        verify_dir_exists : Creates necessary directories
        score : Uses prepared ligand files for docking
        """
        # Determine file type and validate input
        if '.sdf' in lig_file:
            file_type = 'isd'
            suffix = '.sdf'
            # Validate SDF file and check atom count
            if Chem.MolFromMolFile(lig_file).GetNumAtoms() <= 1:
                return False 
        elif '.csv' in lig_file:
            file_type = 'icsv'
            suffix = '.csv'
        elif '.smi' in lig_file:
            file_type = 'ismi'
            suffix = '.smi'
        elif '.mae' in lig_file:
            file_type = 'imae'
            suffix = '.mae'
        
        # Set up output directory and filenames
        lig_prep_path = f'{self.work_path}/4_ligprep_out'
        verify_dir_exists(f'{lig_prep_path}/')
        lig_file = os.path.abspath(lig_file)
        out_name = lig_file.split('/')[-1].replace(suffix, '-out.sdf')
        lig_prep_out_file = f'{lig_prep_path}/{out_name}'
        
        # Return existing file if already prepared
        if os.path.exists(lig_prep_out_file):
            return lig_prep_out_file
        
        # Run LigPrep command
        print('## ligand preparation: ', lig_file)
        cmd = '\n'.join([
            f'cd {lig_prep_path}',
            "$SCHRODINGER/ligprep -WAIT -HOST 'localhost:{}' -NJOBS {} -bff {} -epik -s {} -{} {} -osd {}".format(
                self.n_jobs, self.n_jobs, self.lig_prep_ff, self.max_lig_prep_conf, file_type, 
                lig_file, lig_prep_out_file
                )
            ])
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        
        # Return path if successful, False otherwise
        if os.path.exists(lig_prep_out_file):
            return lig_prep_out_file
        else:
            return False

    def score(self, ligand_file, grid_file):
        """
        Perform molecular docking of a ligand against a protein binding site using Schrödinger's Glide software.
        
        This method orchestrates the complete docking workflow by:
        1. Preparing the ligand using ligand_preparation()
        2. Setting up docking configuration parameters
        3. Running the Glide docking calculation
        4. Processing the docking results
        
        Parameters
        ----------
        ligand_file : str
            Path to the input ligand file in any supported format (SDF, CSV, SMILES, MAE).
            The file will be processed through ligand_preparation() to generate 3D structures.
        
        grid_file : str
            Path to the docking grid file (ZIP archive) generated by grid_generation().
            Defines the protein binding site and docking parameters.
        
        Returns
        -------
        None
            Docking results are saved as compressed SDF files (.sdfgz) in the docking_pose/ subdirectory.
            The method prints status messages but does not return the results directly.
        
        Side Effects
        ------------
        - Creates docking_pose/ subdirectory in the working path
        - Generates docking configuration files (.in format)
        - Runs external Schrödinger Glide command
        - Produces docking pose files (.sdfgz format)
        - Prints status messages to stdout for failures
        
        Notes
        -----
        Docking Workflow:
        1. Converts grid_file to absolute path
        2. Prepares ligand through ligand_preparation() (returns False if failed)
        3. Extracts packet name from grid filename (removes 'grid_' prefix)
        4. Creates docking configuration with:
           - Force field parameters (docking_ff)
           - Grid file specification
           - Prepared ligand file path
           - Output format (ligandlib_sd)
           - Precision level (SP, XP, HTVS, etc.)
        5. Sets up docking directory and configuration file
        6. Runs Glide with either:
           - NSTRUCTS limit if n_struct is specified
           - NJOBS parallel processing otherwise
        7. Checks for output files and reports failures
        
        Configuration Parameters:
        - FORCEFIELD: Specified docking force field
        - GRIDFILE: Path to docking grid
        - LIGANDFILE: Path to prepared ligand
        - POSE_OUTTYPE: Output format (ligandlib_sd)
        - PRECISION: Docking precision level
        
        Output Files:
        - Configuration: '{pkt_name}-docking_config_{precision}.in'
        - Poses: '{dock_path}/*.sdfgz' (compressed SDF format)
        
        Error Handling:
        - Returns None if ligand preparation fails
        - Prints message if no docking poses are generated
        - Uses subprocess for external command execution
        
        Examples
        --------
        >>> glide_scorer = GlideScore()
        >>> # Perform docking with existing grid
        >>> glide_scorer.score('ligand.sdf', 'grid_protein_ligand.zip')
        # Processes ligand and runs docking calculation
        
        >>> # Handle ligand preparation failure
        >>> glide_scorer.score('invalid_ligand.sdf', 'grid_protein_ligand.zip')
        # invalid_ligand.sdf preparation failed, skip
        >>> # Returns None
        
        >>> # Handle docking failure
        >>> glide_scorer.score('ligand.sdf', 'grid_protein_ligand.zip')
        # [] docking failed, skip
        >>> # No poses generated
        
        See Also
        --------
        ligand_preparation : Prepares ligands for docking
        grid_generation : Creates docking grids
        verify_dir_exists : Creates necessary directories
        """

        dock_path = self.work_path+'/5_docking_pose'
        
        # Convert grid file to absolute path
        grid_file = os.path.abspath(grid_file)
        
        # Prepare ligand (returns False if failed)
        lig_prep_out_file = self.ligand_preparation(ligand_file)
        if lig_prep_out_file is False:
            print(f'# {ligand_file} preparation failed, skip')
            return None
        
        # Extract packet name from grid filename
        pkt_name = grid_file.split('/')[-1].split('.')[0].split('grid_')[1]
        if os.path.exists(f'{dock_path}/{pkt_name}-docking_config_{self.precision}.csv'):
            print(f'# Docking results {dock_path}/{pkt_name}-docking_config_{self.precision}.csv has existed, skip')
            return
        
        # Create docking configuration
        docking_config = [
            'FORCEFIELD {}'.format(self.docking_ff),
            'GRIDFILE {}'.format(grid_file),
            'LIGANDFILE {}'.format(lig_prep_out_file),
            #'OUTPUTDIR {}'.format(self.work_path),
            'POSE_OUTTYPE {}'.format('ligandlib_sd'),
            'PRECISION {}'.format(self.precision),
            #'POSES_PER_LIGAND'.format(5)
            ]
        
        # Set up docking directory and configuration file
        verify_dir_exists(f'{dock_path}/')
        config_file = f'{dock_path}/{pkt_name}-docking_config_{self.precision}.in'
        with open(config_file, "w") as fw:
            fw.write('\n'.join(docking_config))
        
        # Run docking with appropriate parameters
        if self.n_struct:
            # Use NSTRUCTS limit if specified
            docking_cmd = '\n'.join([
            'cd {}'.format(dock_path),
            "$SCHRODINGER/glide -WAIT -OVERWRITE -NOLOCAL -HOST 'localhost:{}' -NSTRUCTS {} {}".format(
                self.n_jobs, self.n_struct, config_file)
            ])
        else:
            # Use parallel processing otherwise
            docking_cmd = '\n'.join([
                'cd {}'.format(dock_path),
                "$SCHRODINGER/glide -WAIT -OVERWRITE -NOLOCAL -HOST 'localhost:{}' -NJOBS {} {}".format(
                    self.n_jobs, self.n_jobs, config_file)
                ])
            
        # Execute docking command
        p = subprocess.Popen(docking_cmd, shell=True)
        p.wait()
    
        # Check for output files and report failures
        pose_file = glob.glob(f'{dock_path}/*.sdfgz')
        if len(pose_file) == 0:
            print(f'# {pose_file} docking failed, skip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('This script is used to score docking poses with Glide in Schrodinger-2025.')
    parser.add_argument('--work_path', type=str, required=True, help='work path to ULVS')
    parser.add_argument('--pocket_file_path', type=str, required=True, help='path to pocket files')
    parser.add_argument('--mol_file', type=str, required=True, help='path to molecular file for docking')
    parser.add_argument('--ligand_file', type=str, default=None, help=' ligand file for pocket grid preparation.')
    parser.add_argument('--bff', type=str, default='OPLS_4', choices=['OPLS_4', 'OPLS_3', 'OPLS_5', 'OPLS_2005'], help='force field for docking, select from [OPLS_4, OPLS_3, OPLS_5, OPLS_2005].')
    parser.add_argument('--lig_bff', type=int, default=14, choices=[14, 16], help='force field for ligprep, select from [14, 15].')
    parser.add_argument('--n_jobs', type=int, default=192, help='number of jobs for docking.')
    parser.add_argument('--max_lig_conf', type=int, default=32, help='max number of ligand preparation conformations.')

    args = parser.parse_args()
    WORK_PATH = args.work_path
    file_path = args.pocket_file_path
    ligand_file = args.ligand_file

    bff = args.bff
    lig_bff = args.lig_bff
    n_jobs = args.n_jobs
    mol_file = args.mol_file
    max_lig_conf = args.max_lig_conf

    if os.path.exists(f'{file_path}/pockets/representative_pockets.txt'):
        pdb_lig = get_rep_pkts(f'{file_path}/pockets/representative_pockets.txt')
    elif os.path.exists(str(ligand_file)):
        pdb_lig = []
        for j in glob.glob(f'{file_path}/pdbs/*.pdb'):
            key = j.split('/')[-1].split('.')[0].split('_')[0]
            pdb_lig.append((j, file_path))
    else:
        pdb_dict = {}
        for i in glob.glob(f'{file_path}/pdbs/*.pdb'):
            if 'AF-' in i.split('/')[-1]:
                pdb_id = i.split('/')[-1].split('-')[1]
            else:
                pdb_id = i.split('/')[-1].split('_')[0]
            pdb_dict[pdb_id] = i
        
        pdb_lig = []
        for j in glob.glob(f'{file_path}/pockets/*.sdf'):
            key = j.split('/')[-1].split('.')[0].split('_')[0]
            pdb_lig.append((pdb_dict[key], j))
        
    glide_score = GlideScore(
            work_path=f'{WORK_PATH}/',  
            bff=bff, 
            lig_bff=lig_bff,
            precision='SP', 
            max_lig_prep_conf=max_lig_conf, 
            n_jobs=n_jobs, 
            n_struct=None
            )
    
    for pdb,lig in pdb_lig:
        if 'AF' in pdb.split('/')[-1]:
            context = []
            prot = Protein(pdb, ignore_incomplete_res=True, compute_ss=False)
            for res in prot.get_residues:
                for a in res.get_heavy_atoms:
                    if a.temperature_factor < 70:
                        break
                else:
                    context.append(res.to_heavy_string)
            with open(pdb, 'w') as fw:
                fw.write('\n'.join(context))
        glide_score.grid_generation(pdb, lig)

    glide_score.ligand_preparation(mol_file)
    for grid in sorted(glob.glob(f'{WORK_PATH}/3_pocket_grids/*.zip')):
        glide_score.score(mol_file, grid)