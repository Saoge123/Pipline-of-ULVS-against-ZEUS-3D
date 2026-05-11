
import csv
import random

class ParseClusterResult(object):
    """
    Class to process molecular clustering results and select representative molecules
    from each cluster for further analysis or reranking.
    """
    def __init__(self, rep_type='first'): 
        """
        Initialize the ParseClusterResult object.
        
        Args:
            #cluster_result (str): Path to the cluster result CSV file
            #filtered_csv (str): Path to the filtered CSV file containing molecular data
            rep_type (str): Strategy for selecting representative molecules ('first' or 'max')
            top_n (int or str): Number of top molecules to select ('all' or positive integer)
            
        Raises:
            ValueError: If rep_type is not 'first' or 'max', or if top_n is invalid
        """

        #self.cluster_result = cluster_result
        #self.filtered_csv = filtered_csv

        # Validate rep_type parameter
        if rep_type.lower() not in ['first', 'max']:
            raise ValueError("rep_type must be 'first' or 'max'")
        self.rep_type = rep_type.lower()

    def parse_filter_result(self, filtered_csv):
        """
        Parse the filtered CSV file and create a dictionary mapping molecule IDs to their data.
        
        Args:
            filtered_csv (str): Path to the filtered CSV file
            
        Returns:
            dict: Dictionary with molecule IDs as keys and molecule data as values
        """
        with open(filtered_csv) as fr:
            all_dict = {
                line.strip().split('\t')[1]:line.strip().split('\t') for line in fr.readlines() if line.strip()
                }
        return all_dict
    
    def parse_cluster_result(self, cluster_result):
        """
        Parse the cluster result CSV file and organize molecules by cluster.
        
        Args:
            cluster_result (str): Path to the cluster result CSV file
            
        Returns:
            dict: Dictionary with cluster indices as keys and lists of (molecule_id, score) tuples as values
        """
        with open(cluster_result) as fr:
            reader = csv.reader(fr, quotechar='"', delimiter=',')
            cluster_dict = {}
            # Skip header line and process each data line
            for items in list(reader)[1:]:
                #if line.strip():
                #items = line.strip().split(',')
                cluster_idx = int(items[0])
                # Group molecules by cluster index
                if cluster_idx in cluster_dict:
                    cluster_dict[cluster_idx].append((items[1], float(items[-1])))
                else:
                    cluster_dict[cluster_idx] = [(items[1], float(items[2]))]
        return cluster_dict
    
    def get_rep_mols(self, cluster_dict):
        """
        Select representative molecules from each cluster based on the specified strategy.
        
        Args:
            rep_type (str): Strategy for selection ('first' or 'max')
            cluster_dict (dict): Dictionary of clusters with molecule data
            
        Returns:
            list: List of representative molecules sorted by score in descending order
        """
        if self.rep_type == 'first':
            # Select the first molecule in each cluster
            l_rep = []
            for cluster_id in cluster_dict:
                l_rep.append(cluster_dict[cluster_id][0]) 
            # Sort by score in descending order
            l_rep = sorted(l_rep, key=lambda x: x[1], reverse=True)
        elif self.rep_type == 'max':
            # Select the molecule with maximum score in each cluster
            l_rep = []
            for cluster_id in cluster_dict:
                l_rep.append(max(cluster_dict[cluster_id], key=lambda x: x[1])) 
            # Sort by score in descending order
            l_rep = sorted(l_rep, key=lambda x: x[1], reverse=True)
        return l_rep

    def __call__(self, filtered_csv, cluster_result):
        """
        Execute the complete pipeline to select molecules for reranking.
        
        Returns:
            list: Final list of selected molecules for reranking
        """
        # Parse the filtered molecular data
        all_dict = self.parse_filter_result(filtered_csv)
        # Parse the clustering results
        cluster_dict = self.parse_cluster_result(cluster_result)
        # Select representative molecules from each cluster
        l_rep = self.get_rep_mols(cluster_dict)
        # Get final list of molecules for reranking
        #rep_top = self.get_mols_for_reranking(l_rep, all_dict)
        rep_top = [all_dict[i[0]] for i in l_rep]
        return rep_top