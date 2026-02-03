#!/usr/bin/env python3
"""
NCBI Data Fetcher for SloGPT - Fixed Version

Fetches genomic and clinical data from NCBI and creates standardized datasets.
"""

import argparse
import requests
import json
import time
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import subprocess


class NCBICrawler:
    """NCBI data fetcher for genomic and clinical datasets."""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.session = requests.Session()
        self.output_dir = Path("datasets")
        self.output_dir.mkdir(exist_ok=True)
        
        # NCBI database endpoints
        self.databases = {
            "pubmed": f"{self.base_url}/efetch.fcgi?db=pubmed&retmode=json",
            "gene": f"{self.base_url}/efetch.fcgi?db=gene&retmode=json",
            "snp": f"{self.base_url}/efetch.fcgi?db=snp&retmode=json",
            "protein": f"{self.base_url}/efetch.fcgi?db=protein&retmode=json",
            "structure": f"{self.base_url}/efetch.fcgi?db=structure&retmode=json",
            "taxonomy": f"{self.base_url}/efetch.fcgi?db=taxonomy&retmode=json",
            "sra": f"{self.base_url}/efetch.fcgi?db=sra&retmode=json",
            "biosample": f"{self.base_url}/efetch.fcgi?db=biosample&retmode=json",
            "clinvar": f"{self.base_url}/efetch.fcgi?db=clinvar&retmode=json",
            "gds": f"{self.base_url}/efetch.fcgi?db=gds&retmode=json",
            "pmc": f"{self.base_url}/efetch.fcgi?db=pmc&retmode=json"
        }
    
    def fetch_database_data(self, database: str, query: str, **kwargs) -> Dict:
        """Fetch data from NCBI database."""
        endpoint = self.databases.get(database)
        if not endpoint:
            raise ValueError(f"Unsupported database: {database}")
        
        params = {
            "db": database,
            "retmode": "json",
            "retmax": kwargs.get("max_records", 1000),
            "term": query
        }
        
        if "organism" in kwargs:
            params["organism"] = kwargs["organism"]
        if "species" in kwargs:
            params["id"] = ",".join(kwargs["species"])
        if "data_type" in kwargs:
            params["data_type"] = kwargs["data_type"]
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Fetched {len(data.get('esearchresult', []))} records from {database}")
                return data
            else:
                print(f"❌ Error fetching from {database}: HTTP {response.status_code}")
                return {"esearchresult": [], "error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return {"esearchresult": [], "error": str(e)}
        except Exception as e:
            return {"esearchresult": [], "error": f"Unexpected error: {e}"}
            
    def get_genomic_sequences(self, query: str, species: str = None, max_sequences: int = 100) -> List[Dict]:
        """Fetch genomic sequences from NCBI."""
        print(f"🧬 Fetching genomic sequences for: {query}")
        
        # Build search query for genomic data
        search_terms = query.split()
        search_query = f"{query}[All Fields]"
        
        return self.fetch_database_data("gene", search_query, max_records=max_sequences, species=species)
    
    def get_gene_expression(self, genes: List[str], organism: str = "human", **kwargs) -> List[Dict]:
        """Fetch gene expression data."""
        print(f"📊 Fetching gene expression for {len(genes)} genes")
        
        # Build query for specific genes
        gene_ids = ",".join(genes)
        
        return self.fetch_database_data("snp", gene_ids, organism=organism, report="gene_expression")
    
    def get_clinical_trials(self, study_id: str, **kwargs) -> List[Dict]:
        """Fetch clinical trial data."""
        print(f"🏥 Fetching clinical trials for study: {study_id}")
        
        params = {
            "db": "clinvar",
            "study_id": study_id,
            "retmode": "json",
            "retmax": kwargs.get("max_records", 500)
        }
        
        return self.fetch_database_data("clinvar", params)
    
    def get_protein_structure(self, pdb_id: str, **kwargs) -> Dict:
        """Fetch protein structure data."""
        print(f"🧪 Fetching protein structure for: {pdb_id}")
        
        params = {
            "db": "protein",
            "id": pdb_id
        }
        
        return self.fetch_database_data("protein", params)
    
    def get_sra_study(self, sra_id: str, **kwargs) -> List[Dict]:
        """Fetch SRA study data."""
        print(f"🧬 Fetching SRA study: {sra_id}")
        
        params = {
            "db": "sra",
            "id": sra_id
        }
        
        return self.fetch_database_data("sra", params)
    
    def get_biosamples(self, biosample: str = None, **kwargs) -> List[Dict]:
        """Fetch biosample data."""
        print(f"� Fetching biosamples: {biosample}")
        
        params = {
            "retmode": "json",
            "retmax": kwargs.get("max_records", 100)
        }
        
        if biosample:
            params["biosample"] = biosample
        
        return self.fetch_database_data("biosample", params)
    
    def get_gds_records(self, query: str, **kwargs) -> List[Dict]:
        """Fetch GDS records."""
        print(f"📋 Fetching GDS records for: {query}")
        
        return self.fetch_database_data("gds", {"term": query, "retmode": "json", "retmax": kwargs.get("max_records", 100)})
    
    def get_pmc_records(self, query: str, **kwargs) -> List[Dict]:
        """Fetch PMC records."""
        print(f"📚 Fetching PMC records for: {query}")
        
        return self.fetch_database_data("pmc", {"term": query, "retmode": "json", "retmax": kwargs.get("max_records", 500)})
    
    def create_dataset(self, data: List[Dict], dataset_name: str, data_type: str) -> str:
        """Create dataset from fetched data."""
        print(f"📦 Creating dataset {dataset_name} from {len(data)} records")
        
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Generate input text from data
        input_text = ""
        for record in data:
            if data_type == "gene_sequences":
                if "esearchresult" in record:
                    features = record.get("feature", [])
                    sequence = record.get("sequence", "")
                    if features:
                        features_str = ", ".join([f.get(f) for f in features])
                        input_text += f">{features_str}\n{sequence}\n\n"
            elif data_type == "gene_expression":
                if "esearchresult" in record:
                    exp_data = record.get("expression_data", {})
                    if exp_data:
                        conditions = exp_data.get("conditions", [])
                        condition_str = ", ".join([c.get("condition", "") for c in conditions])
                        value = exp_data.get("value", "")
                        gene_name = exp_data.get("gene", "")
                        input_text += f"{gene_name}\n{condition_str}\n{value}\n\n"
            else:
                # Generic record handling
                record_text = json.dumps(record, indent=2)
                input_text += f"{record_text}\n\n"
        
        # Create input file
        input_file = dataset_dir / "input.txt"
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(input_text)
        
        # Use existing dataset creation tools
        cmd = f"python3 create_dataset_fixed.py {dataset_name} \"{input_text[:5000]}...\""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Dataset {dataset_name} created successfully!")
            return dataset_name
        else:
            print(f"❌ Failed to create dataset: {result.stderr}")
            return None
    
    def get_database_info(self) -> Dict:
        """Get information about available databases."""
        info = {
            "available_databases": list(self.databases.keys()),
            "base_url": self.base_url,
            "description": "NCBI E-utilities for genomic and clinical data access",
            "supported_formats": ["JSON", "FASTA", "XML"],
            "rate_limits": {
                "pubmed": "100 requests/minute",
                "gene": "1000 requests/hour",
                "snp": "1000 requests/hour",
                "protein": "1000 requests/hour"
            }
        }
        return info


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NCBI Genomic Data Crawler for SloGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
             # Search gene sequences
             python ncbi_crawler.py gene "BRCA1" --species human,mouse --max_sequences 50
             
             # Fetch gene expression
             python ncbi_crawler.py expression TP53 --genes "BRCA1,TP53" --organism human --output tp53_expression
             
             # Fetch clinical trials
             python ncbi_crawler.py clinical SRP123456 --max_records 100
             
             # Create dataset
             python ncbi_crawler.py create mygenomic_dataset --query "human cancer" --data_type gene_sequences
         """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize crawler
    crawler = NCBICrawler()
    
    # Search commands
    gene_parser = subparsers.add_parser('gene', help='Search gene sequences')
    gene_parser.add_argument('--query', required=True, help='Search term')
    gene_parser.add_argument('--species', help='Species filter (comma-separated)')
    gene_parser.add_argument('--max_sequences', type=int, default=100, help='Maximum sequences to fetch')
    gene_parser.add_argument('--data_type', choices=['gene_sequences', 'gene_expression'], default='gene_sequences')
    
    # Expression commands
    expression_parser = subparsers.add_parser('expression', help='Fetch gene expression')
    expression_parser.add_argument('--genes', required=True, help='Gene list (comma-separated)')
    expression_parser.add_argument('--organism', help='Organism filter')
    expression_parser.add_argument('--species', help='Target organism')
    
    # Clinical commands  
    clinical_parser = subparsers.add_parser('clinical', help='Fetch clinical trials')
    clinical_parser.add_argument('--study_id', required=True, help='Study ID')
    clinical_parser.add_argument('--max_records', type=int, default=500, help='Maximum records')
    
    # Protein commands
    protein_parser = subparsers.add_parser('protein', help='Fetch protein structure')
    protein_parser.add_argument('--pdb_id', required=True, help='PDB ID')
    
    # Structure commands
    structure_parser = subparsers.add_parser('structure', help='Fetch protein 3D structure')
    structure_parser.add_argument('--pdb_id', required=True, help='PDB ID')
    
    # SRA commands
    sra_parser = subparsers.add_parser('sra', help='Fetch SRA studies')
    sra_parser.add_argument('--sra_id', required=True, help='SRA ID')
    sra_parser.add_argument('--max_records', type=int, default=100, help='Maximum records')
    
    # BioSample commands
    biosample_parser = subparsers.add_parser('biosample', help='Fetch biosamples')
    biosample_parser.add_argument('--biosample', help='Biosample name')
    
    # GDS commands
    gds_parser = subparsers.add_parser('gds', help='Fetch GDS records')
    gds_parser.add_argument('--query', required=True, help='Search term')
    gds_parser.add_argument('--max_records', type=int, default=100, help='Maximum records')
    
    # PMC commands
    pmc_parser = subparsers.add_parser('pmc', help='Fetch PMC records')
    pmc_parser.add_argument('--query', required=True, help='Search term')
    pmc_parser.add_argument('--max_records', type=int, default=100, help='Maximum records')
    
    # Dataset creation
    create_parser = subparsers.add_parser('create', help='Create dataset from fetched data')
    create_parser.add_argument('name', required=True, help='Dataset name')
    create_parser.add_argument('--query', help='Search term for dataset')
    create_parser.add_argument('--data_type', choices=['gene_sequences', 'gene_expression', 'clinical_trials', 'protein_structures', 'sra_studies', 'biosamples'], default='gene_sequences')
    
    args = parser.parse_args()
    
    # Execute appropriate command
    if args.command == 'create':
        if not args.query:
            print("❌ Error: --query required for dataset creation")
            parser.print_help()
            return
        
        data_type = args.data_type or 'gene_sequences'
        
        # Create dataset from fetched data
        if data_type == 'gene_sequences':
            data = args.query or "human genes"
            dataset_name = args.name or f"genomic_sequences_{int(time.time())}"
        elif data_type == 'gene_expression':
            genes = args.genes or "TP53,BRCA1" 
            organism = args.organism or "human"
            dataset_name = args.name or f"gene_expression_{int(time.time())}"
        else:
            data = args.query
        
        # Fetch data first, then create dataset
        print(f"🔍 Fetching {data_type} data...")
        try:
            if data_type == 'gene_sequences':
                data = crawler.get_genomic_sequences(data, args.species, args.max_sequences)
            elif data_type == 'gene_expression':
                data = crawler.get_gene_expression(genes, organism)
            elif data_type == 'clinical_trials':
                data = crawler.get_clinical_trials(args.study_id, args.max_records)
            elif data_type == 'protein_structures':
                data = crawler.get_protein_structure(args.pdb_id)
            elif data_type == 'sra_studies':
                data = crawler.get_sra_study(args.sra_id, args.max_records)
            elif data_type == 'biosamples':
                data = crawler.get_biosamples(args.biosample)
            elif data_type == 'gds':
                data = crawler.get_gds_records(args.query, args.max_records)
            elif data_type == 'pmc':
                data = crawler.get_pmc_records(args.query, args.max_records)
            else:
                print(f"❌ Unsupported data type: {data_type}")
                parser.print_help()
                return
        
            if data:
                dataset_name = crawler.create_dataset(data, dataset_name, data_type)
                print(f"✅ Successfully created dataset: {dataset_name}")
                print(f"🎯 Train with: python3 train_simple.py {dataset_name}")
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return
    
    elif args.command in ['gene', 'expression', 'clinical', 'protein', 'structure', 'sra', 'biosample', 'gds', 'pmc']:
        # Handle data fetching commands
        print(f"🔍 Fetching {args.command} data...")
        
        try:
            if args.command == 'gene':
                data = crawler.get_genomic_sequences(args.query, args.species, args.max_sequences)
            elif args.command == 'expression':
                data = crawler.get_gene_expression(args.genes, args.organism)
            elif args.command == 'clinical':
                data = crawler.get_clinical_trials(args.study_id, args.max_records)
            elif args.command == 'protein':
                data = crawler.get_protein_structure(args.pdb_id)
            elif args.command == 'structure':
                data = crawler.get_protein_structure(args.pdb_id)
            elif args.command == 'sra':
                data = crawler.get_sra_study(args.sra_id, args.max_records)
            elif args.command == 'biosample':
                data = crawler.get_biosamples(args.biosample)
            elif args.command == 'gds':
                data = crawler.get_gds_records(args.query, args.max_records)
            elif args.command == 'pmc':
                data = crawler.get_pmc_records(args.query, args.max_records)
            else:
                print(f"❌ Unknown command: {args.command}")
                parser.print_help()
                return
        
            if data:
                print(f"✅ Successfully fetched {len(data.get('esearchresult', []))} {args.command} records")
                
                # Save raw data for inspection
                data_file = crawler.output_dir / f"{args.command}_raw_data.json"
                with open(data_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"📊 Raw data saved to: {data_file}")
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return
            
    elif args.command == 'info':
        info = crawler.get_database_info()
        print("📊 NCBI Database Info:")
        for db_name in info["available_databases"]:
            print(f"  - {db_name}")
        print(f"    - Base URL: {info['base_url']}")
        print(f"    - Rate limits: {info['rate_limits']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()