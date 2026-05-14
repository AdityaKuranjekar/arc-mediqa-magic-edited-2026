#!/usr/bin/env python3
"""
Diagnosis-based RAG Medical Analysis Pipeline
Advanced agentic system with knowledge retrieval, self-reflection, and multi-cycle reasoning.
"""

import os
import glob
import pandas as pd
import ast
import re
from collections import defaultdict
import json
import datetime
import time
import traceback
from PIL import Image
from dotenv import load_dotenv
from google import genai
import numpy as np
import lancedb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import torch
import random
from typing import Dict, List, Tuple, Optional, Any


class Args:
    """Configuration arguments for the RAG pipeline."""
    
    def __init__(self, use_finetuning=True, use_test_dataset=True, base_dir=None, output_dir=None,
                 model_predictions_dir=None, images_dir=None, dataset_path=None, gemini_model=None,
                 max_reflection_cycles=None, confidence_threshold=None, knowledge_db_path=None,
                 embedding_model=None, cross_encoder_model=None, vector_dimension=None,
                 top_k_semantic=None, top_k_keyword=None, top_k_hybrid=None, top_k_rerank=None,
                 dataset_name_huggingface=None, question_type_retrieval_config=None,
                 default_rag_config=None,
                 fast_triage_confidence_threshold=None, enforce_modality_separation=None,
                 model_name=None):
        """
        Initialize arguments with options for dataset and model type.
        
        Parameters:
        - use_finetuning: Whether to use the fine-tuned model predictions (True) or base model predictions (False)
        - use_test_dataset: Whether to use the test dataset (True) or validation dataset (False)
        - base_dir: Base directory for the project (defaults to current working directory)
        - output_dir: Output directory for results (defaults to base_dir/outputs)
        - model_predictions_dir: Directory containing model predictions (defaults to output_dir)
        - images_dir: Directory containing images (auto-determined based on dataset if not provided)
        - dataset_path: Path to dataset CSV file (auto-determined based on dataset if not provided)
        - gemini_model: Gemini model to use (defaults to gemini-2.0-flash-exp-2025-01-29)
        - max_reflection_cycles: Maximum number of reflection cycles (defaults to 2)
        - confidence_threshold: Confidence threshold for reflection (defaults to 0.75)
        - knowledge_db_path: Path to knowledge database (defaults to base_dir/knowledge_db)
        - embedding_model: Embedding model for semantic search (defaults to BioBERT)
        - cross_encoder_model: Cross-encoder model for reranking (defaults to ms-marco-MiniLM)
        - vector_dimension: Vector dimension for embeddings (defaults to 768)
        - top_k_semantic: Top K results for semantic search (defaults to 7)
        - top_k_keyword: Top K results for keyword search (defaults to 7)
        - top_k_hybrid: Top K results for hybrid search (defaults to 10)
        - top_k_rerank: Top K results after reranking (defaults to 5)
        - dataset_name_huggingface: HuggingFace dataset name (defaults to Skin_diseases_and_care)
        - question_type_retrieval_config: Configuration for question type retrieval (defaults to predefined config)
        - default_rag_config: Default RAG configuration (defaults to {"use_rag": True, "weight": 0.4})
        - fast_triage_confidence_threshold: Confidence threshold above which Fast Triage Agent
          bypasses the full pipeline (Gap 1). Defaults to 0.95.
        - enforce_modality_separation: If True, Clinical Context Agent and Image Analysis Agent
          operate strictly on their own modality only (Gap 2). Defaults to True.
        """
        self.use_finetuning = use_finetuning
        self.use_test_dataset = use_test_dataset
        
        # Set base directory
        self.base_dir = base_dir or os.getcwd()
        
        # Set output directory
        self.output_dir = output_dir or os.path.join(self.base_dir, "outputs")
        
        # Set model predictions directory
        self.model_predictions_dir = model_predictions_dir or self.output_dir
        
        # Set dataset-specific configurations
        if self.use_test_dataset:
            self.dataset_name = "test"
            self.dataset_path = dataset_path or os.path.join(self.output_dir, "test_dataset.csv")
            self.images_dir = images_dir or os.path.join(self.base_dir, "2025_dataset", "test", "images_test")
            self.prediction_prefix = "test_aggregated_predictions_"
        else:
            self.dataset_name = "validation"
            self.dataset_path = dataset_path or os.path.join(self.output_dir, "val_dataset.csv")
            self.images_dir = images_dir or os.path.join(self.base_dir, "2025_dataset", "valid", "images_valid")
            self.prediction_prefix = "val_aggregated_predictions_"
        
        self.model_name = model_name
        self.model_type = "finetuned" if self.use_finetuning else "base"
        
        # Model and processing configuration
        self.gemini_model = gemini_model or "gemini-2.0-flash"
        self.max_reflection_cycles = max_reflection_cycles or 2
        self.confidence_threshold = confidence_threshold or 0.75
        
        # Knowledge base configuration
        self.knowledge_db_path = knowledge_db_path or os.path.join(self.base_dir, "knowledge_db")
        self.embedding_model = embedding_model or "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        self.cross_encoder_model = cross_encoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.vector_dimension = vector_dimension or 768
        self.top_k_semantic = top_k_semantic or 7
        self.top_k_keyword = top_k_keyword or 7
        self.top_k_hybrid = top_k_hybrid or 10
        self.top_k_rerank = top_k_rerank or 5
        
        self.dataset_name_huggingface = dataset_name_huggingface or "brucewayne0459/Skin_diseases_and_care"
        
        # Question type retrieval configuration
        self.question_type_retrieval_config = question_type_retrieval_config or {
            "Site Location": {"use_rag": False, "weight": 0.2},
            "Lesion Color": {"use_rag": False, "weight": 0.2},
            "Size": {"use_rag": False, "weight": 0.1},
            "Skin Description": {"use_rag": True, "weight": 0.3},
            "Onset": {"use_rag": True, "weight": 0.4},
            "Itch": {"use_rag": True, "weight": 0.4},
            "Extent": {"use_rag": False, "weight": 0.2},
            "Treatment": {"use_rag": True, "weight": 0.7},
            "Lesion Evolution": {"use_rag": True, "weight": 0.5},
            "Texture": {"use_rag": True, "weight": 0.3},
            "Lesion Count": {"use_rag": False, "weight": 0.1},
            "Differential": {"use_rag": True, "weight": 0.8},
            "Specific Diagnosis": {"use_rag": True, "weight": 0.8},
        }
        
        self.default_rag_config = default_rag_config or {"use_rag": True, "weight": 0.4}

        # Gap 1: Fast Triage Gatekeeper threshold
        self.fast_triage_confidence_threshold = (
            fast_triage_confidence_threshold
            if fast_triage_confidence_threshold is not None else 0.95
        )

        # Gap 2: Asymmetric Partitioning enforcement flag
        self.enforce_modality_separation = (
            enforce_modality_separation
            if enforce_modality_separation is not None else True
        )

        print(f"\nRAG Pipeline Configuration initialized:")
        print(f"- Base directory: {self.base_dir}")
        print(f"- Output directory: {self.output_dir}")
        print(f"- Using {'test' if self.use_test_dataset else 'validation'} dataset")
        print(f"- Looking for {self.model_type} model predictions")
        print(f"- Dataset path: {self.dataset_path}")
        print(f"- Images directory: {self.images_dir}")
        print(f"- Model predictions directory: {self.model_predictions_dir}")
        print(f"- Prediction file prefix: {self.prediction_prefix}")
        print(f"- Gemini model: {self.gemini_model}")
        print(f"- Knowledge DB path: {self.knowledge_db_path}")
        print(f"- Max reflection cycles: {self.max_reflection_cycles}")
        print(f"- Confidence threshold: {self.confidence_threshold}")
        print(f"- Fast triage threshold (Gap 1): {self.fast_triage_confidence_threshold}")
        print(f"- Enforce modality separation (Gap 2): {self.enforce_modality_separation}")


class DataLoader:
    """Handles loading and processing of model predictions and validation data."""
    
    @staticmethod
    def get_latest_aggregated_files(args):
        """Get the latest aggregated prediction files for each model."""
        # Use a more flexible pattern that matches the pipeline.py output format
        pattern = os.path.join(args.model_predictions_dir, f"{args.prediction_prefix}*.csv")
        
        agg_files = glob.glob(pattern)
        
        if len(agg_files) == 0:
            return []
        
        latest_files = {}
        
        for file_path in agg_files:
            file_name = os.path.basename(file_path)
            
            model_part = file_name.replace(args.prediction_prefix, "").replace(".csv", "")
            
            # The model name is everything before the timestamp
            # Format: {prefix}{model_name}_{timestamp}.csv
            parts = model_part.rsplit('_', 1)
            if len(parts) != 2:
                print(f"Warning: Unexpected filename format (no timestamp?): {file_name}")
                continue
            
            model_name = parts[0]
            
            # Filter by model_name if provided
            if args.model_name and model_name != args.model_name:
                continue

            try:
                timestamp = int(parts[1])
            except ValueError:
                print(f"Warning: Could not parse timestamp in {file_name}")
                continue
            
            if model_name not in latest_files or timestamp > latest_files[model_name]['timestamp']:
                latest_files[model_name] = {
                    'file_path': file_path,
                    'timestamp': timestamp
                }
        
        return [info['file_path'] for _, info in latest_files.items()]
    
    @staticmethod
    def load_all_model_predictions(args):
        """Load all model predictions from aggregated files."""
        latest_files = DataLoader.get_latest_aggregated_files(args)
        
        if not latest_files:
            print("No aggregated prediction files found. Cannot proceed.")
            return {}
        
        model_predictions = {}
        
        for file_path in latest_files:
            file_name = os.path.basename(file_path)
            
            model_part = file_name.replace(args.prediction_prefix, "").replace(".csv", "")
            model_name = model_part.rsplit('_', 1)[0]
            
            try:
                df = pd.read_csv(file_path)
                df['model_name'] = model_name
                model_predictions[model_name] = df
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return model_predictions

    @staticmethod
    def load_validation_dataset(args):
        """Load the validation dataset."""
        val_df = pd.read_csv(args.dataset_path)
        
        val_df = DataLoader.process_validation_dataset(val_df)
        
        encounter_question_data = defaultdict(lambda: {
            'images': [],
            'data': None
        })
        
        for _, row in val_df.iterrows():
            encounter_id = row['encounter_id']
            base_qid = row['base_qid']
            key = (encounter_id, base_qid)
            
            if 'image_path' in row and row['image_path']:
                encounter_question_data[key]['images'].append(row['image_path'])
            elif 'image_id' in row and row['image_id']:
                image_path = os.path.join(args.images_dir, row['image_id'])
                encounter_question_data[key]['images'].append(image_path)
            
            if encounter_question_data[key]['data'] is None:
                encounter_question_data[key]['data'] = row.to_dict()
        
        grouped_data = []
        for (encounter_id, base_qid), data in encounter_question_data.items():
            entry = data['data'].copy()
            entry['all_images'] = data['images']
            entry['encounter_id'] = encounter_id
            entry['base_qid'] = base_qid
            grouped_data.append(entry)
        
        return pd.DataFrame(grouped_data)
    
    @staticmethod
    def safe_convert_options(options_str):
        """Safely convert a string representation of a list to an actual list."""
        if not isinstance(options_str, str):
            return options_str
            
        try:
            return ast.literal_eval(options_str)
        except (SyntaxError, ValueError):
            if options_str.startswith('[') and options_str.endswith(']'):
                return [opt.strip().strip("'\"") for opt in options_str[1:-1].split(',')]
            elif ',' in options_str:
                return [opt.strip() for opt in options_str.split(',')]
            else:
                return [options_str]
    
    @staticmethod
    def process_validation_dataset(val_df):
        """Process and clean the validation dataset."""
        if 'options_en' in val_df.columns:
            val_df['options_en'] = val_df['options_en'].apply(DataLoader.safe_convert_options)
            
            def clean_options(options):
                if not isinstance(options, list):
                    return options
                    
                cleaned_options = []
                for opt in options:
                    if isinstance(opt, str):
                        cleaned_opt = opt.strip("'\" ").replace(" (please specify)", "")
                        cleaned_options.append(cleaned_opt)
                    else:
                        cleaned_options.append(str(opt).strip("'\" "))
                return cleaned_options
                
            val_df['options_en_cleaned'] = val_df['options_en'].apply(clean_options)
        
        if 'question_text' in val_df.columns:
            val_df['question_text_cleaned'] = val_df['question_text'].apply(
                lambda q: q.replace(" Please specify which affected area for each selection.", "") 
                          if isinstance(q, str) and "Please specify which affected area for each selection" in q 
                          else q
            )
            
            val_df['question_text_cleaned'] = val_df['question_text_cleaned'].apply(
                lambda q: re.sub(r'^\d+\s+', '', q) if isinstance(q, str) else q
            )
        
        if 'base_qid' not in val_df.columns and 'qid' in val_df.columns:
            val_df['base_qid'] = val_df['qid'].apply(
                lambda q: q.split('-')[0] if isinstance(q, str) and '-' in q else q
            )
        
        return val_df


class DataProcessor:
    """Handles data processing for query creation."""
    
    @staticmethod
    def create_query_context(row, args=None):
        """Create query context from validation data similar to the inference process."""
        question = row.get('question_text_cleaned', row.get('question_text', 'What do you see in this image?'))
        
        metadata = ""
        if 'question_type_en' in row:
            metadata += f"Type: {row['question_type_en']}"
            
        if 'question_category_en' in row:
            metadata += f", Category: {row['question_category_en']}"
        
        query_title = row.get('query_title_en', '')
        query_content = row.get('query_content_en', '')
        
        clinical_context = ""
        if query_title or query_content:
            clinical_context += "Background Clinical Information (to help with your analysis):\n"
            if query_title:
                clinical_context += f"{query_title}\n"
            if query_content:
                clinical_context += f"{query_content}\n"
        
        options = row.get('options_en_cleaned', row.get('options_en', ['Yes', 'No', 'Not mentioned']))
        if isinstance(options, list):
            options_text = ", ".join(options)
        else:
            options_text = str(options)
        
        query_text = (f"MAIN QUESTION TO ANSWER: {question}\n"
                     f"Question Metadata: {metadata}\n"
                     f"{clinical_context}"
                     f"Available Options (choose from these): {options_text}")
        
        return query_text


class AgenticRAGData:
    """Manages combined data for agentic reasoning with RAG."""
    
    def __init__(self, all_models_df, validation_df):
        self.all_models_df = all_models_df
        self.validation_df = validation_df
        
        self.model_predictions = {}
        for (encounter_id, base_qid), group in all_models_df.groupby(['encounter_id', 'base_qid']):
            self.model_predictions[(encounter_id, base_qid)] = group
        
        self.validation_data = {}
        for _, row in validation_df.iterrows():
            self.validation_data[(row['encounter_id'], row['base_qid'])] = row
    
    def get_combined_data(self, encounter_id, base_qid):
        """Retrieve combined data for a specific encounter and question."""
        model_preds = self.model_predictions.get((encounter_id, base_qid), None)
        
        val_data = self.validation_data.get((encounter_id, base_qid), None)
        
        if model_preds is None:
            print(f"No model predictions found for encounter {encounter_id}, question {base_qid}")
            return None
            
        if val_data is None:
            print(f"No validation data found for encounter {encounter_id}, question {base_qid}")
            return None
        
        if 'query_context' not in val_data:
            val_data['query_context'] = DataProcessor.create_query_context(val_data)
        
        model_predictions_dict = {}
        for _, row in model_preds.iterrows():
            model_name = row['model_name']
            
            model_predictions_dict[model_name] = self._process_model_predictions(row)
        
        return {
            'encounter_id': encounter_id,
            'base_qid': base_qid,
            'query_context': val_data['query_context'],
            'images': val_data.get('all_images', []),
            'options': val_data.get('options_en_cleaned', val_data.get('options_en', [])),
            'question_type': val_data.get('question_type_en', ''),
            'question_category': val_data.get('question_category_en', ''),
            'model_predictions': model_predictions_dict
        }
    
    def _process_model_predictions(self, row):
        """Process model predictions from row data."""
        return {
            'model_prediction': row.get('combined_prediction', '')
        }
    
    def get_all_encounter_question_pairs(self):
        """Return a list of all unique encounter_id, base_qid pairs."""
        return list(self.validation_data.keys())
    
    def get_sample_data(self, n=5):
        """Get a sample of combined data for n random encounter-question pairs."""
        import random
        
        all_pairs = self.get_all_encounter_question_pairs()
        sample_pairs = random.sample(all_pairs, min(n, len(all_pairs)))
        
        return [self.get_combined_data(encounter_id, base_qid) for encounter_id, base_qid in sample_pairs]


def parse_json_response(text):
    """Parse JSON from LLM response."""
    cleaned_text = text
    if "```json" in cleaned_text:
        cleaned_text = cleaned_text.split("```json")[1]
    if "```" in cleaned_text:
        cleaned_text = cleaned_text.split("```")[0]
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse as JSON")
        return {"parse_error": "Could not parse as JSON", "raw_text": text}


class KnowledgeBaseManager:
    """Manages the dermatology knowledge base for RAG."""

    def __init__(self, args=None):
        """Initialize the knowledge base manager."""
        self.args = args
        self.embedding_model = SentenceTransformer(args.embedding_model if args else "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        self.cross_encoder = CrossEncoder(args.cross_encoder_model if args else "cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.db_path = args.knowledge_db_path if args else os.path.join(os.getcwd(), "knowledge_db")
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)

        self.table_name = "dermatology_knowledge"

        if self.table_name not in self.db.table_names():
            print(f"Knowledge base not found. Creating new knowledge base at {self.db_path}")
            self._initialize_knowledge_base()
        else:
            print(f"Using existing knowledge base at {self.db_path}")
            self.table = self.db.open_table(self.table_name)

        self.tokenized_corpus = []
        self.doc_ids = []
        self._initialize_bm25_index()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with the skin diseases dataset."""
        print("Loading dermatology dataset...")
        dataset_name = self.args.dataset_name_huggingface if self.args else "brucewayne0459/Skin_diseases_and_care"
        dataset = load_dataset(dataset_name)

        data = []

        print("Processing dataset and creating embeddings...")
        for i, item in enumerate(dataset['train']):
            topic = item['Topic']
            information = item['Information']

            combined_text = f"Topic: {topic}\n\nInformation: {information}"

            embedding = self.embedding_model.encode(combined_text)

            data.append({
                "id": i,
                "topic": topic,
                "information": information,
                "combined_text": combined_text,
                "vector": embedding.tolist()
            })

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} documents")

        import pandas as pd
        data_df = pd.DataFrame(data)

        print("Creating vector database...")
        self.table = self.db.create_table(
            self.table_name,
            data=data_df
        )
        print("Knowledge base initialization complete.")
    
    def _initialize_bm25_index(self):
        """Initialize the BM25 index for keyword search without NLTK dependencies."""
        print("Initializing BM25 index...")

        results = self.table.search().limit(10000).to_pandas()

        common_stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
            "by", "about", "from", "as", "of", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "can", "could", "will",
            "would", "shall", "should", "may", "might", "must", "this", "that", "these",
            "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her"
        }

        for idx, row in results.iterrows():
            doc_text = row['combined_text']
            self.doc_ids.append(row['id'])

            tokens = []
            for token in doc_text.lower().split():
                token = ''.join(c for c in token if c.isalnum())
                if token and token not in common_stopwords:
                    tokens.append(token)

            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index initialization complete.")
    
    def semantic_search(self, query, top_k=None):
        """Perform semantic search using embeddings."""
        if top_k is None:
            top_k = self.args.top_k_semantic if self.args else 7
        
        query_embedding = self.embedding_model.encode(query)
        
        results = self.table.search(query_embedding.tolist()).limit(top_k).to_pandas()
        
        return results
    
    def keyword_search(self, query, top_k=None):
        """Perform keyword search using BM25."""
        if top_k is None:
            top_k = self.args.top_k_keyword if self.args else 7

        common_stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
            "by", "about", "from", "as", "of", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "can", "could", "will",
            "would", "shall", "should", "may", "might", "must", "this", "that", "these",
            "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her"
        }

        query_tokens = []
        for token in query.lower().split():
            token = ''.join(c for c in token if c.isalnum())
            if token and token not in common_stopwords:
                query_tokens.append(token)

        doc_scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                doc_id = self.doc_ids[idx]
                score = doc_scores[idx]
                
                doc = self.table.search().where(f"id = {doc_id}").limit(1).to_pandas()
                
                if not doc.empty:
                    results.append({
                        "id": doc_id,
                        "topic": doc['topic'].iloc[0],
                        "information": doc['information'].iloc[0],
                        "combined_text": doc['combined_text'].iloc[0],
                        "_distance": 1.0 - min(score / 10.0, 1.0)
                    })
        
        return pd.DataFrame(results)
    
    def hybrid_search(self, query, top_k=None):
        """Perform hybrid search combining semantic and keyword search."""
        if top_k is None:
            top_k = self.args.top_k_hybrid if self.args else 10
        
        semantic_results = self.semantic_search(query, top_k=top_k)
        keyword_results = self.keyword_search(query, top_k=top_k)
        
        combined_results = pd.concat([semantic_results, keyword_results])
        combined_results = combined_results.drop_duplicates(subset=['id'])
        
        if len(combined_results) > 0:
            return self.rerank_results(combined_results, query, top_k=min(top_k, len(combined_results)))
        else:
            return pd.DataFrame()
    
    def rerank_results(self, results, query, top_k=None):
        """Rerank search results using a cross-encoder."""
        if top_k is None:
            top_k = self.args.top_k_rerank if self.args else 5
        
        if len(results) == 0:
            return pd.DataFrame()
        
        pairs = [(query, doc) for doc in results['combined_text'].tolist()]
        
        cross_scores = self.cross_encoder.predict(pairs)
        
        results = results.copy()
        results['cross_score'] = cross_scores
        
        results = results.sort_values(by='cross_score', ascending=False).head(top_k)
        
        return results


class DiagnosisExtractor:
    """Extracts potential diagnoses from image analysis and clinical context."""
    
    @staticmethod
    def extract_diagnoses(image_analysis, clinical_context, query_context=None, args=None):
        """
        Extract potential diagnoses from image analysis, clinical context, and query.
        
        Args:
            image_analysis: Structured image analysis containing OVERALL_IMPRESSION
            clinical_context: Structured clinical context analysis
            query_context: Optional clinical query side feed
            
        Returns:
            List of dictionaries with diagnoses and confidence scores
        """
        diagnoses = []
        
        if image_analysis and "aggregated_analysis" in image_analysis:
            if "OVERALL_IMPRESSION" in image_analysis["aggregated_analysis"]:
                impression = image_analysis["aggregated_analysis"]["OVERALL_IMPRESSION"]
                if isinstance(impression, str):
                    diagnoses.extend(DiagnosisExtractor._extract_from_text(impression, source="image_analysis", confidence=0.7))
        
        if clinical_context and "structured_clinical_context" in clinical_context:
            if "DIAGNOSTIC_CONSIDERATIONS" in clinical_context["structured_clinical_context"]:
                diagnostic_info = clinical_context["structured_clinical_context"]["DIAGNOSTIC_CONSIDERATIONS"]
                if isinstance(diagnostic_info, str):
                    diagnoses.extend(DiagnosisExtractor._extract_from_text(diagnostic_info, source="clinical_context", confidence=0.6))

        # Handle Query side feed (from Gap 2 diagram)
        if query_context:
            diagnoses.extend(DiagnosisExtractor._extract_from_text(query_context, source="query_context", confidence=0.5))

        if not diagnoses:
            diagnoses = DiagnosisExtractor._suggest_from_features(image_analysis, clinical_context)
            
        return diagnoses
    
    @staticmethod
    def _extract_from_text(text, source, confidence):
        """Extract diagnoses from text."""
        import re
        
        diagnostic_terms = [
            "eczema", "dermatitis", "psoriasis", "acne", "rosacea", "urticaria", 
            "melanoma", "carcinoma", "pemphigus", "pemphigoid", "lupus", "scleroderma",
            "folliculitis", "cellulitis", "impetigo", "tinea", "herpes", "wart",
            "vitiligo", "alopecia", "lichen", "keratosis", "prurigo", "rash"
        ]
        
        diagnoses = []
        
        for term in diagnostic_terms:
            pattern = fr'\b({term})[s\s]\b'
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                diagnoses.append({
                    "diagnosis": match.group(1).capitalize(),
                    "confidence": confidence,
                    "source": source
                })
                
        patterns = [
            r'consistent with\s+([^,.;]+)',
            r'suggestive of\s+([^,.;]+)',
            r'indicative of\s+([^,.;]+)',
            r'compatible with\s+([^,.;]+)',
            r'diagnostic of\s+([^,.;]+)',
            r'likely\s+([^,.;]+)',
            r'probable\s+([^,.;]+)',
            r'possible\s+([^,.;]+)',
            r'suspected\s+([^,.;]+)',
            r'diagnosis of\s+([^,.;]+)',
            r'impression:\s+([^,.;]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                diagnoses.append({
                    "diagnosis": match.group(1).strip().capitalize(),
                    "confidence": confidence * 0.9,
                    "source": source
                })
        
        unique_diagnoses = []
        seen = set()
        for diag in diagnoses:
            if diag["diagnosis"].lower() not in seen:
                seen.add(diag["diagnosis"].lower())
                unique_diagnoses.append(diag)
        
        return unique_diagnoses
    
    @staticmethod
    def _suggest_from_features(image_analysis, clinical_context):
        """Suggest potential diagnoses based on extracted features."""
        diagnoses = []
        features = {}
        
        if image_analysis and "aggregated_analysis" in image_analysis:
            analysis = image_analysis["aggregated_analysis"]
            
            if "SKIN_DESCRIPTION" in analysis:
                features["skin_description"] = analysis["SKIN_DESCRIPTION"]
                
            if "LESION_COLOR" in analysis:
                features["lesion_color"] = analysis["LESION_COLOR"]
                
            if "SITE_LOCATION" in analysis:
                features["site_location"] = analysis["SITE_LOCATION"]
        
        if clinical_context and "structured_clinical_context" in clinical_context:
            context = clinical_context["structured_clinical_context"]
            
            if "SYMPTOMS" in context:
                features["symptoms"] = context["SYMPTOMS"]
                
            if "HISTORY" in context:
                features["history"] = context["HISTORY"]
        
        if features:
            if "hand" in str(features.get("site_location", "")).lower():
                if "scaling" in str(features.get("skin_description", "")).lower():
                    diagnoses.append({
                        "diagnosis": "Hand eczema",
                        "confidence": 0.5,
                        "source": "feature_based"
                    })
                    
            if "red" in str(features.get("lesion_color", "")).lower():
                if "itchy" in str(features.get("symptoms", "")).lower():
                    diagnoses.append({
                        "diagnosis": "Contact dermatitis",
                        "confidence": 0.4,
                        "source": "feature_based"
                    })
        
        if not diagnoses:
            diagnoses.append({
                "diagnosis": "Dermatosis", 
                "confidence": 0.3,
                "source": "fallback"
            })
            
        return diagnoses


class DiagnosisBasedQueryGenerator:
    """Generates search queries based on extracted diagnoses."""
        
    def __init__(self, client, args=None):
        """Initialize the query generator."""
        self.client = client
        self.args = args
    
    def generate_queries(self, question_text, question_type, options, integrated_evidence, diagnoses, num_queries=4):
        """
        Generate search queries based on diagnoses and question type.
        
        Args:
            question_text: The question text
            question_type: Type of question being asked
            options: Available answer options
            integrated_evidence: Integrated evidence from images and clinical context
            diagnoses: List of extracted diagnoses
            num_queries: Number of queries to generate
            
        Returns:
            List of search queries
        """
        sorted_diagnoses = sorted(diagnoses, key=lambda x: x.get('confidence', 0), reverse=True)
        
        question_specific_queries = self._generate_question_specific_queries(
            question_text, 
            question_type, 
            options, 
            sorted_diagnoses
        )
        
        diagnosis_specific_queries = self._generate_diagnosis_specific_queries(
            question_type,
            sorted_diagnoses
        )
        
        all_queries = question_specific_queries + diagnosis_specific_queries
        
        unique_queries = []
        seen = set()
        for query in all_queries:
            if query.lower() not in seen:
                seen.add(query.lower())
                unique_queries.append(query)
        
        return unique_queries[:num_queries]
    
    def _generate_question_specific_queries(self, question_text, question_type, options, diagnoses):
        """Generate queries specific to the question type."""
        queries = []
        
        classification_types = ["Site Location", "Lesion Color", "Size", "Extent", "Lesion Count"]
        if question_type in classification_types:
            classification_terms = ", ".join([opt for opt in options if opt.lower() != "not mentioned"])
            queries.append(f"dermatology {question_type.lower()} classification {classification_terms}")
            
            if len(options) > 2:
                queries.append(f"how to distinguish between {classification_terms} in dermatology")
                
            if question_type == "Extent":
                queries.append("definition of widespread vs limited area skin condition dermatology")
        
        if question_type in ["Differential", "Specific Diagnosis"]:
            if diagnoses:
                top_diagnosis = diagnoses[0]["diagnosis"]
                queries.append(f"{top_diagnosis} diagnostic criteria dermatology")
                
                diagnoses_list = ", ".join([d["diagnosis"] for d in diagnoses[:3]])
                queries.append(f"differential diagnosis {diagnoses_list}")
        
        if question_type == "Treatment":
            if diagnoses:
                top_diagnosis = diagnoses[0]["diagnosis"]
                queries.append(f"{top_diagnosis} treatment options dermatology")
                
                body_site = self._extract_body_site(question_text)
                if body_site:
                    queries.append(f"{top_diagnosis} {body_site} treatment guidelines")
        
        return queries
    
    def _generate_diagnosis_specific_queries(self, question_type, diagnoses):
        """Generate queries that connect diagnoses with the question type."""
        queries = []
        
        if not diagnoses:
            return queries
            
        for diagnosis in diagnoses[:2]:
            diag_name = diagnosis["diagnosis"]
            
            if question_type in ["Site Location", "Extent"]:
                queries.append(f"{diag_name} typical distribution pattern dermatology")
                queries.append(f"{diag_name} localized versus widespread presentation")
                
            elif question_type == "Lesion Color":
                queries.append(f"{diag_name} typical color appearance dermatology")
                
            elif question_type == "Texture":
                queries.append(f"{diag_name} texture characteristics dermatology")
                
            elif question_type == "Itch":
                queries.append(f"is {diag_name} itchy dermatology")
                
            elif question_type == "Onset":
                queries.append(f"{diag_name} typical onset and progression")
                
            else:
                queries.append(f"{diag_name} {question_type.lower()} dermatology")
                
        return queries
    
    def _extract_body_site(self, question_text):
        """Extract body site from question text."""
        import re
        
        body_parts = [
            "hand", "foot", "arm", "leg", "face", "back", "chest", "abdomen",
            "scalp", "neck", "finger", "toe", "elbow", "knee", "shoulder",
            "palm", "sole", "trunk", "extremity", "head"
        ]
        
        for part in body_parts:
            if re.search(r'\b' + part + r'[s]?\b', question_text.lower()):
                return part
                
        return None


class DiagnosisBasedKnowledgeRetriever:
    """Retrieves knowledge from the dermatology knowledge base using diagnosis-based approach."""
    
    def __init__(self, kb_manager, query_generator, diagnosis_extractor, args=None):
        """
        Initialize the knowledge retriever.
        
        Args:
            kb_manager: KnowledgeBaseManager instance
            query_generator: DiagnosisBasedQueryGenerator instance
            diagnosis_extractor: DiagnosisExtractor instance
            args: Configuration arguments
        """
        self.kb_manager = kb_manager
        self.query_generator = query_generator
        self.diagnosis_extractor = diagnosis_extractor
        self.args = args
    
    def retrieve_knowledge(self, question_text, question_type, options, image_analysis, clinical_context, integrated_evidence):
        """
        Retrieve relevant knowledge for a dermatological question using diagnoses.
        
        Args:
            question_text: The question text
            question_type: Type of question being asked
            options: Available answer options
            image_analysis: Structured image analysis
            clinical_context: Structured clinical context
            integrated_evidence: Integrated evidence from images and clinical context
            
        Returns:
            Dictionary with retrieved knowledge
        """
        if self.args:
            rag_config = self.args.question_type_retrieval_config.get(
                question_type, self.args.default_rag_config
            )
        else:
            default_config = {"use_rag": True, "weight": 0.4}
            rag_config = {
                "Site Location": {"use_rag": False, "weight": 0.2},
                "Lesion Color": {"use_rag": False, "weight": 0.2},
                "Size": {"use_rag": False, "weight": 0.1},
            }.get(question_type, default_config)
        
        if not rag_config["use_rag"]:
            return {
                "retrieved": False,
                "reason": f"RAG not enabled for question type: {question_type}",
                "results": []
            }
        
        diagnoses = self.diagnosis_extractor.extract_diagnoses(image_analysis, clinical_context, question_text)
        
        queries = self.query_generator.generate_queries(
            question_text, 
            question_type, 
            options, 
            integrated_evidence,
            diagnoses
        )
        
        if not queries:
            return {
                "retrieved": False,
                "reason": "Failed to generate search queries",
                "results": []
            }
        
        all_results = []
        
        for query in queries:
            results = self.kb_manager.hybrid_search(query)
            
            if not results.empty:
                for _, row in results.iterrows():
                    relevance_score = float(row.get('cross_score', 1.0 - row.get('_distance', 0.5)))
                    
                    if relevance_score > 0:
                        all_results.append({
                            "query": query,
                            "topic": row['topic'],
                            "information": row['information'],
                            "relevance_score": relevance_score,
                            "diagnoses": [d["diagnosis"] for d in diagnoses[:3]]
                        })
        
        unique_results = []
        seen_topics = set()
        
        for result in sorted(all_results, key=lambda x: x['relevance_score'], reverse=True):
            if result['topic'] not in seen_topics:
                unique_results.append(result)
                seen_topics.add(result['topic'])
        
        top_k = self.args.top_k_rerank if self.args else 5
        
        return {
            "retrieved": len(unique_results) > 0,
            "queries": queries,
            "diagnoses": diagnoses,
            "results": unique_results[:top_k]
        }


class AgenticDermatologyPipeline:
    """Main pipeline for agentic dermatology analysis with diagnosis-based retrieval."""
    
    def __init__(self, api_key=None, args=None):
        if api_key is None:
            api_key = os.getenv("API_KEY")
        
        self.client = genai.Client(api_key=api_key)
        self.args = args
        
        print("Initializing knowledge base...")
        self.kb_manager = KnowledgeBaseManager(args)
        
        self.diagnosis_extractor = DiagnosisExtractor()
        
        self.query_generator = DiagnosisBasedQueryGenerator(self.client, args)
        
        self.knowledge_retriever = DiagnosisBasedKnowledgeRetriever(
            self.kb_manager,
            self.query_generator,
            self.diagnosis_extractor,
            args
        )
    
    def _call_gemini_with_retry(self, model, contents, config=None, max_retries=5):
        """Call Gemini API with fallback for 404s and retry for 429s."""
        import time as _time
        import random as _random

        # --- Resilience Upgrade ---
        PRIMARY_MODEL = model
        FALLBACK_MODEL = "gemini-2.5-flash"  # Synced with discovery list
        
        current_model = PRIMARY_MODEL
        base_delay = 5          # Reduced base delay for paid tier
        
        for attempt in range(max_retries + 1):
            try:
                kwargs = {"model": current_model, "contents": contents}
                if config:
                    kwargs["config"] = config
                response = self.client.models.generate_content(**kwargs)
                return response

            except Exception as exc:
                error_str = str(exc).lower()
                
                # Handle 404: Try fallback model immediately
                if "404" in error_str or "not_found" in error_str:
                    if current_model != FALLBACK_MODEL:
                        print(f"  [Resilience] 404 on '{current_model}'. Trying fallback '{FALLBACK_MODEL}'...")
                        current_model = FALLBACK_MODEL
                        # Retry immediately in the next iteration
                        continue 
                    else:
                        raise exc

                is_rate_limit = (
                    "429" in error_str
                    or "resource_exhausted" in error_str
                    or "quota" in error_str
                    or "rate limit" in error_str
                )
                if is_rate_limit and attempt < max_retries:
                    # Exponential backoff with ±10 % jitter
                    delay = base_delay * (2 ** attempt) * (1 + _random.uniform(-0.1, 0.1))
                    print(
                        f"  [Rate Limit] 429/RESOURCE_EXHAUSTED on attempt "
                        f"{attempt + 1}/{max_retries}. "
                        f"Backing off {delay:.1f} s before retry..."
                    )
                    _time.sleep(delay)
                else:
                    # Non-rate-limit error, or retries exhausted → propagate
                    if attempt >= max_retries:
                        print(
                            f"  [Gemini] All {max_retries} retries exhausted. "
                            f"Last error: {exc}"
                        )
                    raise exc
    
    def fast_triage(self, patient_description: str, images: List[str], query_context: str) -> Dict[str, Any]:
        """
        Phase 1 — Fast Triage Gatekeeper (Gap 1).

        All three inputs (description, images, query) converge here before any heavy pipeline
        is triggered. If confidence exceeds 0.95, bypass=True is returned and process_question
        routes the answer directly to Final Diagnosis, skipping the ensemble, RAG retrieval,
        and synthesis chain entirely.

        Args:
            patient_description: Raw patient-authored text description.
            images: List of image file paths for the encounter.
            query_context: Full query string including question, options, and metadata.

        Returns:
            Dict with keys:
                - answer (str): Preliminary answer if high confidence, else empty string.
                - confidence (float): Confidence score 0–1.
                - bypass (bool): True if confidence > 0.95 (skip downstream agents).
        """
        prompt = f"""You are a clinical triage assistant performing a rapid first-pass assessment
of a dermatological case before any detailed analysis pipeline is engaged.

PATIENT DESCRIPTION:
{patient_description}

CLINICAL QUERY:
{query_context}

Your task:
1. Review the patient description and any images provided.
2. Attempt to answer the clinical query using only what is immediately visible and described.
3. Assign a confidence score (0.0 to 1.0) reflecting how certain you are of your answer.

IMPORTANT RULE:
- If you are highly confident (confidence > 0.95) — meaning the answer is unambiguous from
  the available information alone — provide a definitive answer and set confidence accordingly.
- If there is ANY ambiguity, uncertainty, or need for deeper multi-modal analysis, set
  confidence BELOW 0.95 so the full pipeline is engaged.
- Do NOT force high confidence. Only report >0.95 when truly certain.

Respond strictly in JSON format:
{{
    "answer": "<your answer here, or empty string if not confident enough>",
    "confidence": <float between 0.0 and 1.0>,
    "bypass": <true if confidence > 0.95, else false>,
    "reasoning": "<brief one-sentence rationale for confidence level>"
}}
"""
        try:
            image_parts = []
            for img_path in images[:5]:
                if os.path.exists(img_path):
                    image_parts.append(Image.open(img_path))

            contents = [prompt] + image_parts if image_parts else [prompt]

            response = self._call_gemini_with_retry(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash",
                contents=contents
            )

            result = parse_json_response(response.text)

            # Enforce type safety and bypass threshold
            confidence = float(result.get("confidence", 0.0))
            answer = str(result.get("answer", ""))
            bypass = confidence > (
                self.args.fast_triage_confidence_threshold
                if self.args and hasattr(self.args, "fast_triage_confidence_threshold")
                else 0.95
            )

            print(f"  [Fast Triage] confidence={confidence:.3f}, bypass={bypass}")

            return {
                "answer": answer,
                "confidence": confidence,
                "bypass": bypass,
                "reasoning": result.get("reasoning", "")
            }

        except Exception as e:
            print(f"  [Fast Triage] Error: {e}. Defaulting to full pipeline (bypass=False).")
            return {"answer": "", "confidence": 0.0, "bypass": False, "reasoning": str(e)}

    def analyze_images(self, images: List[str]) -> Dict[str, Any]:
        """
        Image Analysis Agent (Gap 2 — vision-only, blind to patient text).

        Receives ONLY the patient images. No clinical text, no patient description,
        no query context is passed to this agent. This enforces strict modality
        separation so the agent cannot anchor its visual findings on textual cues.

        Extracts all 10 clinical dimensions described in the paper:
        SIZE, SITE_LOCATION, SKIN_DESCRIPTION, LESION_COLOR, LESION_COUNT,
        EXTENT, TEXTURE, ONSET_INDICATORS, ITCH_INDICATORS, OVERALL_IMPRESSION.

        Args:
            images: List of image file paths for the encounter.

        Returns:
            Dict with per-image observations and aggregated analysis across all images.
        """
        if not images:
            return {"error": "No images provided"}

        prompt = """You are a dermatology image analysis agent. Your task is to analyze the \
provided skin images visually.

STRICT RULE: Base your analysis ONLY on what you can see in the images. Do NOT infer or \
assume anything from clinical descriptions, patient history, or text — none of that information \
is available to you. Analyze purely from visual evidence.

For each image, extract observations across all 10 clinical dimensions. Then produce an \
aggregated analysis that consolidates findings across all images.

Respond strictly in JSON format:
{
    "individual_images": [
        {
            "image_index": 0,
            "observations": {
                "SIZE": "approximate lesion size(s) — e.g. 1-3mm papules, 5cm plaque",
                "SITE_LOCATION": "exact anatomical location(s) visible in this image",
                "SKIN_DESCRIPTION": "morphology — e.g. papules, plaques, vesicles, erosions",
                "LESION_COLOR": "all colors observed — e.g. erythematous, hyperpigmented",
                "LESION_COUNT": "number of discrete lesions visible or 'diffuse/extensive'",
                "EXTENT": "how much body surface is affected in this image — localized or widespread",
                "TEXTURE": "surface texture — e.g. smooth, rough, scaly, crusted, warty",
                "ONSET_INDICATORS": "visual signs suggesting acute vs chronic — e.g. fresh vesicles, lichenification",
                "ITCH_INDICATORS": "visual signs suggesting pruritus — e.g. excoriations, linear scratch marks",
                "OVERALL_IMPRESSION": "clinical impression from visual findings in this image alone"
            }
        }
    ],
    "aggregated_analysis": {
        "SIZE": "size range across all images",
        "SITE_LOCATION": "all anatomical locations combined across images",
        "SKIN_DESCRIPTION": "consolidated morphology across all images",
        "LESION_COLOR": "all colors seen across all images",
        "LESION_COUNT": "total count or pattern across all images",
        "EXTENT": "overall body surface involvement",
        "TEXTURE": "predominant and variant textures across images",
        "ONSET_INDICATORS": "aggregated chronicity indicators",
        "ITCH_INDICATORS": "aggregated pruritus signs",
        "OVERALL_IMPRESSION": "unified clinical impression and most likely condition(s) based purely on visual evidence"
    }
}
"""

        try:
            image_parts = []
            for img_path in images[:5]:
                if os.path.exists(img_path):
                    image_parts.append(Image.open(img_path))

            if not image_parts:
                return {"error": "No valid image files found at provided paths"}

            response = self._call_gemini_with_retry(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash",
                contents=[prompt] + image_parts
            )

            image_analysis_agent_result = parse_json_response(response.text)
            print(f"  [Image Analysis Agent] Analyzed {len(image_parts)} image(s) — vision-only.")
            return image_analysis_agent_result

        except Exception as e:
            print(f"  [Image Analysis Agent] Error: {e}")
            return {"error": str(e)}
    
    def extract_clinical_context(self, patient_description: str) -> Dict[str, Any]:
        """
        Clinical Context Agent (Gap 2 — text-only, blind to images).

        Receives ONLY the patient-authored text description (query_title_en + query_content_en).
        No image paths, no visual observations, no aggregated image findings are passed here.
        This enforces strict modality separation — the agent cannot anchor on visual cues.

        Parses patient text into 13 structured clinical categories from the paper:
        DEMOGRAPHICS, CHIEF_COMPLAINT, SYMPTOM_PROGRESSION, ONSET_DURATION, TRIGGERS,
        AGGRAVATING_FACTORS, RELIEVING_FACTORS, PRIOR_TREATMENTS, MEDICAL_HISTORY,
        FAMILY_HISTORY, SOCIAL_HISTORY, REVIEW_OF_SYSTEMS, DIAGNOSTIC_CONSIDERATIONS.

        Args:
            patient_description: Raw patient-authored text (title + content combined).

        Returns:
            Dict with structured_clinical_context containing 13 categories.
        """
        prompt = f"""You are a clinical context extraction agent. Your task is to parse \
patient-authored text and extract structured clinical information.

STRICT RULE: Base your analysis ONLY on the text provided below. Do NOT reference, infer, \
or assume anything from images or visual descriptions — no images are available to you. \
Work purely from what the patient has written.

If a category is not mentioned in the text, output "Not mentioned" for that field.
Filter out informal language and noise while preserving all clinically relevant details.

PATIENT TEXT:
{patient_description}

Respond strictly in JSON format:
{{
    "structured_clinical_context": {{
        "DEMOGRAPHICS": "age, sex, relevant demographic information if mentioned",
        "CHIEF_COMPLAINT": "the primary skin concern or reason for seeking help",
        "SYMPTOM_PROGRESSION": "how symptoms have changed over time — worsening, stable, improving",
        "ONSET_DURATION": "when symptoms first appeared and how long they have been present",
        "TRIGGERS": "events, substances, or exposures that started or triggered the condition",
        "AGGRAVATING_FACTORS": "what makes symptoms worse",
        "RELIEVING_FACTORS": "what makes symptoms better or provides relief",
        "PRIOR_TREATMENTS": "any treatments attempted before — medications, home remedies, hospital visits",
        "MEDICAL_HISTORY": "relevant past medical conditions, surgeries, or chronic illnesses",
        "FAMILY_HISTORY": "any similar conditions or relevant illnesses in family members",
        "SOCIAL_HISTORY": "occupation, lifestyle, travel, or exposures relevant to the condition",
        "REVIEW_OF_SYSTEMS": "other symptoms mentioned beyond the skin — e.g. fever, fatigue, joint pain",
        "DIAGNOSTIC_CONSIDERATIONS": "potential diagnoses suggested by or inferable from the text alone"
    }}
}}
"""
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.5-flash-preview-04-17",
                contents=prompt
            )

            clinical_context_agent_result = parse_json_response(response.text)
            print(f"  [Clinical Context Agent] Parsed patient description — text-only.")
            return clinical_context_agent_result

        except Exception as e:
            print(f"  [Clinical Context Agent] Error: {e}")
            return {"error": str(e)}
    
    def integrate_evidence(self, image_analysis: Dict, clinical_context: Dict,
                           retrieved_knowledge: Dict, query_context: str) -> Dict[str, Any]:
        """
        Evidence Integration Agent.

        Receives outputs from the Image Analysis Agent, Clinical Context Agent, and
        Knowledge Retrieval Agent (conditional on query type). The Clinical Query is
        passed as a side feed to guide adaptive weighting.

        IMPORTANT: Model predictions are NOT passed here. Per the methodology, model
        predictions feed exclusively into the Asymmetric Synthesizer downstream.

        Applies adaptive task-specific weights:
        - Appearance questions (color, size, texture): emphasise visual cues
        - Treatment/history questions: emphasise clinical context
        - Diagnosis questions: emphasise retrieved literature

        Args:
            image_analysis: Output from Image Analysis Agent (vision-only).
            clinical_context: Output from Clinical Context Agent (text-only).
            retrieved_knowledge: Output from Knowledge Retrieval Agent (may be empty
                                 if RAG is disabled for this question type).
            query_context: The clinical query string — used as a side feed to determine
                           which modality to weight most heavily.

        Returns:
            Dict with integrated_findings containing merged, weighted evidence.
        """
        rag_available = (
            isinstance(retrieved_knowledge, dict)
            and retrieved_knowledge.get("retrieved", False)
            and retrieved_knowledge.get("results")
        )

        retrieved_section = (
            json.dumps(retrieved_knowledge, indent=2)
            if rag_available
            else "Not retrieved for this question type (RAG disabled or no results)."
        )

        prompt = f"""You are the Evidence Integration Agent in a clinical multi-agent pipeline.

Your role is to synthesize outputs from two independent agents — one that analyzed images \
visually, and one that parsed the patient's text description — together with any retrieved \
medical literature. You must NOT generate a final answer here; your job is to produce a \
structured, weighted evidence summary that will be passed to the Reasoning Engine.

CLINICAL QUERY (side feed — use to determine weighting):
{query_context}

WEIGHTING RULES (apply adaptively based on the question type above):
- Visual/appearance questions (color, morphology, texture, size, extent): weight IMAGE ANALYSIS highest
- Clinical history questions (onset, triggers, prior treatment): weight CLINICAL CONTEXT highest
- Diagnosis/differential questions: weight RETRIEVED KNOWLEDGE highest
- For all questions: flag any contradictions between modalities explicitly

---

IMAGE ANALYSIS AGENT OUTPUT (vision-only):
{json.dumps(image_analysis, indent=2)}

CLINICAL CONTEXT AGENT OUTPUT (text-only):
{json.dumps(clinical_context, indent=2)}

RETRIEVED MEDICAL KNOWLEDGE:
{retrieved_section}

---

Respond strictly in JSON format:
{{
    "integrated_findings": {{
        "dominant_modality": "which source (image/clinical/literature) dominates for this question type and why",
        "visual_evidence_summary": "key findings from image analysis relevant to the query",
        "clinical_evidence_summary": "key findings from patient text relevant to the query",
        "literature_evidence_summary": "key points from retrieved knowledge, or 'Not available'",
        "supporting_evidence": "combined evidence supporting the most likely answer(s)",
        "conflicting_evidence": "any contradictions between image findings and patient text, or 'None'",
        "modality_weights": {{
            "image_analysis": "<high/medium/low>",
            "clinical_context": "<high/medium/low>",
            "retrieved_knowledge": "<high/medium/low>"
        }},
        "integrated_hypothesis": "best hypothesis synthesized from all available evidence"
    }}
}}
"""
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.5-flash-preview-04-17",
                contents=prompt
            )

            result = parse_json_response(response.text)
            print(f"  [Evidence Integration Agent] RAG used: {rag_available}.")
            return result

        except Exception as e:
            print(f"  [Evidence Integration Agent] Error: {e}")
            return {"error": str(e)}
    
    def asymmetric_synthesizer(self, integrated_evidence: Dict, model_predictions: Dict,
                                query_context: str, options: List[str]) -> Dict[str, Any]:
        """
        Asymmetric Synthesizer — Reasoning Engine Agent (Gap 2).

        This is the core decision-making agent. It receives:
          - integrated_evidence: weighted synthesis from Evidence Integration Agent
          - model_predictions: advisory inputs from all 7 VLMs (side feed)
          - query_context: the clinical query (side feed for adaptive weighting)
          - options: the valid answer choices to select from

        Model predictions are treated as ADVISORY INPUTS — expert opinions to be
        critically evaluated, not votes to be counted. The agent resolves contradictions
        between modalities mathematically using adaptive weighting rather than consensus.

        The returned confidence score gates the safety loop:
          - confidence >= 0.75 → answer goes directly to Final Diagnosis
          - confidence <  0.75 → routed to Self-Reflection Agent → Re-Analysis Agent

        Args:
            integrated_evidence: Output from Evidence Integration Agent.
            model_predictions: Dict of {model_name: {"model_prediction": str}} from all VLMs.
            query_context: Clinical query string (side feed).
            options: List of valid answer options for this question.

        Returns:
            Dict with answer, confidence (0–1), reasoning, and uncertainty_factors.
        """
        # Format model predictions clearly as advisory inputs
        advisory_block = "\n".join([
            f"  - {name}: {preds.get('model_prediction', 'N/A')}"
            for name, preds in model_predictions.items()
        ]) if model_predictions else "  No model predictions available."

        options_str = ", ".join(f'"{o}"' for o in options)

        prompt = f"""You are the Asymmetric Synthesizer — the final reasoning agent in a \
clinical multi-agent pipeline for dermatological Visual Question Answering.

You receive:
1. A weighted evidence summary produced by the Evidence Integration Agent
2. Predictions from 7 independent vision-language models (advisory inputs only)
3. The clinical query and valid answer options

Your task is to synthesize all inputs and select the best answer(s) from the options provided.

===== CLINICAL QUERY (side feed) =====
{query_context}

===== VALID OPTIONS (select ONLY from these) =====
{options_str}

===== INTEGRATED EVIDENCE (from Evidence Integration Agent) =====
{json.dumps(integrated_evidence, indent=2)}

===== MODEL PREDICTIONS (advisory inputs — 7 VLMs) =====
{advisory_block}

===== SYNTHESIS INSTRUCTIONS =====

STEP 1 — READ THE EVIDENCE:
Identify the dominant_modality and modality_weights from the integrated evidence.
Apply those weights when evaluating support for each option.

STEP 2 — EVALUATE MODEL PREDICTIONS CRITICALLY:
Treat the 7 model predictions as expert opinions, NOT as votes.
- Look for patterns of agreement across models
- Flag any model that contradicts the integrated evidence
- Do not let majority consensus override strong evidence from the dominant modality

STEP 3 — RESOLVE CONTRADICTIONS ASYMMETRICALLY:
If image evidence and clinical text evidence contradict each other:
- For appearance questions (color, morphology, texture, size, extent): trust image evidence
- For history questions (onset, duration, treatment): trust clinical context
- For diagnosis questions: trust retrieved literature evidence
Weight the contradiction toward the modality that the query type favors.

STEP 4 — SELECT ANSWER(S):
- For single-answer questions: select exactly one option
- For multi-label questions: select all options supported by the evidence
- Only select options present in the VALID OPTIONS list above

STEP 5 — ASSIGN CONFIDENCE:
Rate your confidence (0.0–1.0) in the selected answer(s).
Confidence < 0.75 means the safety loop (Self-Reflection → Re-Analysis) will activate.
Be honest — underconfidence triggers a helpful re-check; overconfidence is dangerous.

Respond strictly in JSON format:
{{
    "answer": "<comma-separated answer(s) chosen from the valid options>",
    "confidence": <float 0.0–1.0>,
    "reasoning": "<structured explanation: which evidence drove the decision and why>",
    "uncertainty_factors": ["<list each source of uncertainty>"],
    "model_agreement_summary": "<brief note on how well the 7 models agreed with each other and with your answer>",
    "dominant_evidence_used": "<image_analysis / clinical_context / retrieved_knowledge>"
}}
"""
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.5-flash-preview-04-17",
                contents=prompt
            )

            result = parse_json_response(response.text)

            confidence = float(result.get("confidence", 0.0))
            answer = str(result.get("answer", "Not mentioned"))

            print(f"  [Asymmetric Synthesizer] answer='{answer}', confidence={confidence:.3f}")

            return {
                "answer": answer,
                "confidence": confidence,
                "reasoning": result.get("reasoning", ""),
                "uncertainty_factors": result.get("uncertainty_factors", []),
                "model_agreement_summary": result.get("model_agreement_summary", ""),
                "dominant_evidence_used": result.get("dominant_evidence_used", "")
            }

        except Exception as e:
            print(f"  [Asymmetric Synthesizer] Error: {e}")
            return {
                "answer": "Not mentioned",
                "confidence": 0.0,
                "reasoning": str(e),
                "uncertainty_factors": ["API error"],
                "model_agreement_summary": "",
                "dominant_evidence_used": ""
            }

    def reason_with_reflection(self, question_text: str, options: List[str],
                               integrated_evidence: Dict, cycle: int = 0) -> Dict[str, Any]:
        """Perform reasoning with self-reflection."""
        prompt = f"""
        Question: {question_text}
        Options: {', '.join(options)}
        
        Integrated Evidence:
        {json.dumps(integrated_evidence, indent=2)}
        
        Reasoning Cycle: {cycle + 1}
        
        Analyze this dermatology question step by step:
        
        1. What is the question specifically asking?
        2. What evidence supports each option?
        3. Which option is most strongly supported?
        4. What is your confidence level (0-1)?
        
        Provide your analysis in JSON format:
        {{
            "reasoning": {{
                "question_analysis": "what the question asks",
                "option_evidence": {{
                    "option1": "evidence for/against",
                    "option2": "evidence for/against"
                }},
                "conclusion": "selected answer",
                "confidence": 0.85,
                "areas_of_uncertainty": ["list of uncertainties"]
            }}
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash-exp-2025-01-29",
                contents=prompt
            )
            
            return parse_json_response(response.text)
            
        except Exception as e:
            print(f"Error in reasoning: {e}")
            return {"error": str(e)}
    
    def self_reflect(self, reasoning: Dict, integrated_evidence: Dict) -> Dict[str, Any]:
        """Perform self-reflection on reasoning."""
        prompt = f"""
        Review this reasoning for potential errors or improvements:
        
        Reasoning:
        {json.dumps(reasoning, indent=2)}
        
        Evidence:
        {json.dumps(integrated_evidence, indent=2)}
        
        Identify:
        1. Any logical errors or inconsistencies
        2. Missing considerations
        3. Whether confidence level is appropriate
        4. Suggestions for improvement
        
        Provide reflection in JSON format:
        {{
            "reflection": {{
                "logical_errors": ["list of errors"],
                "missing_considerations": ["what was missed"],
                "confidence_assessment": "is confidence appropriate?",
                "improvement_suggestions": ["suggestions"],
                "should_revise": true/false
            }}
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.0-flash-exp-2025-01-29",
                contents=prompt
            )
            
            return parse_json_response(response.text)
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            return {"error": str(e)}

    def re_analyze(self, reflection_output: Dict, integrated_evidence: Dict,
                   model_predictions: Dict, options: List[str],
                   query_context: str) -> Dict[str, Any]:
        """
        Re-Analysis Agent — Safety Loop Path.

        This is a DISTINCT agent from self_reflect. It activates only when the
        Self-Reflection Agent has identified specific reasoning gaps or errors in
        the Asymmetric Synthesizer's initial output.

        Rather than a generic re-run, this agent conducts a TARGETED deep reassessment,
        focusing specifically on the gaps identified during reflection. It has access to:
          - The reflection's identified errors and suggestions
          - The full integrated evidence
          - All model predictions (for cross-checking)
          - The original query and options

        Its output feeds directly into Final Diagnosis on the safety loop path.

        Args:
            reflection_output: Output from Self-Reflection Agent containing identified gaps.
            integrated_evidence: Output from Evidence Integration Agent.
            model_predictions: Dict of {model_name: {"model_prediction": str}} from all VLMs.
            options: List of valid answer options.
            query_context: Clinical query string.

        Returns:
            Dict with revised answer, new confidence score, and targeted reasoning.
        """
        # Extract identified gaps from reflection
        reflection = reflection_output.get("reflection", {})
        logical_errors = reflection.get("logical_errors", [])
        missing_considerations = reflection.get("missing_considerations", [])
        improvement_suggestions = reflection.get("improvement_suggestions", [])

        gaps_block = (
            f"LOGICAL ERRORS IDENTIFIED:\n" +
            "\n".join(f"  - {e}" for e in logical_errors if e) +
            f"\n\nMISSING CONSIDERATIONS:\n" +
            "\n".join(f"  - {m}" for m in missing_considerations if m) +
            f"\n\nIMPROVEMENT SUGGESTIONS:\n" +
            "\n".join(f"  - {s}" for s in improvement_suggestions if s)
        ) if (logical_errors or missing_considerations or improvement_suggestions) else \
            "No specific gaps identified — conduct a full independent reassessment."

        advisory_block = "\n".join([
            f"  - {name}: {preds.get('model_prediction', 'N/A')}"
            for name, preds in model_predictions.items()
        ]) if model_predictions else "  No model predictions available."

        options_str = ", ".join(f'"{o}"' for o in options)

        prompt = f"""You are the Re-Analysis Agent — the final safety checkpoint in a \
clinical multi-agent pipeline for dermatological Visual Question Answering.

You have been activated because the Asymmetric Synthesizer's initial confidence was below \
0.75. The Self-Reflection Agent has identified specific gaps in the prior reasoning. \
Your task is a TARGETED deep reassessment that directly addresses those gaps — not a \
repeat of the same reasoning.

===== CLINICAL QUERY =====
{query_context}

===== VALID OPTIONS (select ONLY from these) =====
{options_str}

===== GAPS IDENTIFIED BY SELF-REFLECTION AGENT =====
{gaps_block}

===== INTEGRATED EVIDENCE =====
{json.dumps(integrated_evidence, indent=2)}

===== MODEL PREDICTIONS (advisory — for cross-checking) =====
{advisory_block}

===== RE-ANALYSIS INSTRUCTIONS =====

1. ADDRESS THE GAPS FIRST:
   For each identified logical error or missing consideration above, explicitly re-examine
   the evidence and determine whether it changes the answer.

2. CROSS-CHECK WITH MODELS:
   Look specifically at whether any model predictions align with the overlooked evidence.

3. APPLY ASYMMETRIC WEIGHTING:
   - Appearance questions → prioritise image evidence
   - History/treatment questions → prioritise clinical context
   - Diagnosis questions → prioritise retrieved literature

4. PRODUCE A REVISED ANSWER:
   Select the best answer(s) from the valid options list only.
   If your re-analysis does not change the original answer, explicitly state why.

5. ASSIGN REVISED CONFIDENCE:
   This is the final confidence — it will not be checked again.

Respond strictly in JSON format:
{{
    "revised_answer": "<comma-separated answer(s) from valid options>",
    "revised_confidence": <float 0.0–1.0>,
    "gap_resolution": "<how each identified gap was addressed and what changed>",
    "revised_reasoning": "<full reasoning for the revised answer>",
    "answer_changed": <true/false>,
    "change_explanation": "<if answer changed: why; if not: why the original was correct>"
}}
"""
        try:
            response = self.client.models.generate_content(
                model=self.args.gemini_model if self.args else "gemini-2.5-flash-preview-04-17",
                contents=prompt
            )

            result = parse_json_response(response.text)

            revised_answer = str(result.get("revised_answer", "Not mentioned"))
            revised_confidence = float(result.get("revised_confidence", 0.0))
            answer_changed = result.get("answer_changed", False)

            print(f"  [Re-Analysis Agent] revised_answer='{revised_answer}', "
                  f"confidence={revised_confidence:.3f}, changed={answer_changed}")

            return {
                "answer": revised_answer,
                "confidence": revised_confidence,
                "reasoning": result.get("revised_reasoning", ""),
                "gap_resolution": result.get("gap_resolution", ""),
                "answer_changed": answer_changed,
                "change_explanation": result.get("change_explanation", "")
            }

        except Exception as e:
            print(f"  [Re-Analysis Agent] Error: {e}")
            return {
                "answer": "Not mentioned",
                "confidence": 0.0,
                "reasoning": str(e),
                "gap_resolution": "",
                "answer_changed": False,
                "change_explanation": "API error during re-analysis"
            }

    def process_question(self, sample_data: Dict) -> Dict[str, Any]:
        """
        Main orchestration method — routes a single question through the full agent pipeline.

        Implements the 8-stage flow from the methodology:

          Stage 1: Fast Triage Gatekeeper (Gap 1)
                   All 3 inputs converge. If confidence > 0.95 → bypass directly to
                   Final Diagnosis. The heavy pipeline is never triggered.

          Stage 2: Clinical Context Agent (Gap 2 — text-only)
                   Receives patient description ONLY. Blind to images.

          Stage 3: Image Analysis Agent (Gap 2 — vision-only)
                   Receives images ONLY. Blind to patient text.
                   Stages 2 & 3 run independently — neither sees the other's modality.

          Stage 4: Diagnosis Extractor Agent
                   Merges image + clinical reports. Receives query as side feed.
                   Extracts diagnostic hypotheses to drive targeted retrieval.

          Stage 5: Knowledge Retrieval Agent (conditional on question type)
                   Hybrid BM25 + BioBERT retrieval with cross-encoder reranking.
                   Returns empty if RAG is disabled for this question type.

          Stage 6: Evidence Integration Agent
                   Merges image analysis + clinical context + retrieved knowledge + query.
                   Model predictions are NOT passed here.

          Stage 7: Asymmetric Synthesizer (Reasoning Engine)
                   Receives integrated evidence + model predictions (side feed) + query.
                   If confidence >= 0.75 → Final Diagnosis (main path).

          Stage 8: Safety Loop (confidence < 0.75)
                   Self-Reflection Agent identifies gaps in synthesizer output.
                   Re-Analysis Agent conducts targeted deep reassessment → Final Diagnosis.

        Args:
            sample_data: Dict from AgenticRAGData.get_combined_data() containing
                         query_context, images, options, question_type, model_predictions,
                         and patient description fields.

        Returns:
            Dict with final_answer, confidence, reasoning, pipeline_path, and
            intermediate outputs from each agent stage.
        """
        query_context = sample_data['query_context']
        question_type = sample_data.get('question_type', '')
        options = sample_data.get('options', [])
        model_predictions = sample_data.get('model_predictions', {})
        images = sample_data.get('images', [])

        # Extract patient description text only (title + content, no images)
        val_data = sample_data  # alias for clarity
        patient_description = "\n".join(filter(None, [
            str(val_data.get('query_title_en', '')),
            str(val_data.get('query_content_en', ''))
        ])).strip()
        # Fallback: parse from query_context if direct fields not available
        if not patient_description:
            patient_description = query_context

        # Extract question text from query_context
        try:
            question_text = query_context.split("MAIN QUESTION TO ANSWER:")[1].split("\n")[0].strip()
        except IndexError:
            question_text = query_context

        confidence_threshold = (
            self.args.confidence_threshold
            if self.args else 0.75
        )
        fast_triage_threshold = (
            self.args.fast_triage_confidence_threshold
            if self.args and hasattr(self.args, 'fast_triage_confidence_threshold') else 0.95
        )

        # ── STAGE 1: Fast Triage Gatekeeper (Gap 1) ─────────────────────────────
        print(f"\n  [Stage 1] Fast Triage Gatekeeper...")
        triage_result = self.fast_triage(patient_description, images, query_context)

        if triage_result.get("bypass", False):
            print(f"  [Stage 1] HIGH CONFIDENCE bypass → Final Diagnosis (skipping full pipeline).")
            return {
                "question_text": question_text,
                "question_type": question_type,
                "options": options,
                "final_answer": triage_result["answer"],
                "confidence": triage_result["confidence"],
                "pipeline_path": "fast_triage_bypass",
                "triage_result": triage_result,
                "image_analysis": {},
                "clinical_context": {},
                "retrieved_knowledge": {},
                "integrated_evidence": {},
                "synthesizer_output": {},
                "reflection": {},
                "re_analysis": {}
            }

        # ── STAGE 2: Clinical Context Agent (text-only, blind to images) ─────────
        print(f"  [Stage 2] Clinical Context Agent (text-only)...")
        clinical_context = self.extract_clinical_context(patient_description)

        # ── STAGE 3: Image Analysis Agent (vision-only, blind to text) ──────────
        print(f"  [Stage 3] Image Analysis Agent (vision-only)...")
        image_analysis = self.analyze_images(images)

        # ── STAGE 4: Diagnosis Extractor (merges both + query side feed) ─────────
        print(f"  [Stage 4] Diagnosis Extractor Agent...")
        diagnoses = self.diagnosis_extractor.extract_diagnoses(image_analysis, clinical_context, query_context)
        print(f"  [Stage 4] Extracted {len(diagnoses)} diagnostic hypothesis/es.")

        # ── STAGE 5: Knowledge Retrieval (conditional on question type) ──────────
        print(f"  [Stage 5] Knowledge Retrieval Agent (question_type='{question_type}')...")
        retrieved_knowledge = self.knowledge_retriever.retrieve_knowledge(
            question_text, question_type, options,
            image_analysis, clinical_context,
            {}  # integrated_evidence not yet available at this stage
        )
        rag_used = retrieved_knowledge.get("retrieved", False)
        print(f"  [Stage 5] RAG retrieved: {rag_used} "
              f"({len(retrieved_knowledge.get('results', []))} results).")

        # ── STAGE 6: Evidence Integration (NO model_predictions) ─────────────────
        print(f"  [Stage 6] Evidence Integration Agent...")
        integrated_evidence = self.integrate_evidence(
            image_analysis,
            clinical_context,
            retrieved_knowledge,
            query_context          # query as side feed only
        )

        # ── STAGE 7: Asymmetric Synthesizer (model_predictions as side feed) ─────
        print(f"  [Stage 7] Asymmetric Synthesizer...")
        synthesizer_output = self.asymmetric_synthesizer(
            integrated_evidence,
            model_predictions,     # advisory side feed — not in evidence integration
            query_context,
            options
        )

        final_answer = synthesizer_output.get("answer", "Not mentioned")
        confidence = synthesizer_output.get("confidence", 0.0)

        # ── STAGE 8: Safety Loop (confidence < threshold) ────────────────────────
        reflection = {}
        re_analysis = {}
        pipeline_path = "main_path"

        if confidence < confidence_threshold:
            print(f"  [Stage 8] Confidence {confidence:.3f} < {confidence_threshold} "
                  f"→ Self-Reflection Agent activated.")
            reflection = self.self_reflect(synthesizer_output, integrated_evidence)

            should_revise = reflection.get("reflection", {}).get("should_revise", True)
            print(f"  [Stage 8] Self-Reflection says should_revise={should_revise} "
                  f"→ Re-Analysis Agent activated.")

            re_analysis = self.re_analyze(
                reflection,
                integrated_evidence,
                model_predictions,
                options,
                query_context
            )

            final_answer = re_analysis.get("answer", final_answer)
            confidence = re_analysis.get("confidence", confidence)
            pipeline_path = "safety_loop_path"
            print(f"  [Stage 8] Re-Analysis final_answer='{final_answer}', "
                  f"confidence={confidence:.3f}.")
        else:
            print(f"  [Stage 7] Confidence {confidence:.3f} >= {confidence_threshold} "
                  f"→ Main path to Final Diagnosis.")

        return {
            "question_text": question_text,
            "question_type": question_type,
            "options": options,
            "final_answer": final_answer,
            "confidence": confidence,
            "pipeline_path": pipeline_path,
            "triage_result": triage_result,
            "image_analysis": image_analysis,
            "clinical_context": clinical_context,
            "retrieved_knowledge": retrieved_knowledge,
            "integrated_evidence": integrated_evidence,
            "synthesizer_output": synthesizer_output,
            "reflection": reflection,
            "re_analysis": re_analysis
        }
    
    def process_single_encounter(self, agentic_data, encounter_id):
        """
        Process a single encounter with all its questions using the agentic pipeline.

        Args:
            agentic_data: AgenticRAGData instance containing all encounter data
            encounter_id: The specific encounter ID to process

        Returns:
            Dictionary with all questions processed with agentic reasoning for this encounter
        """
        all_pairs = agentic_data.get_all_encounter_question_pairs()
        encounter_pairs = [pair for pair in all_pairs if pair[0] == encounter_id]

        if not encounter_pairs:
            print(f"No data found for encounter {encounter_id}")
            return None

        print(f"Processing {len(encounter_pairs)} questions for encounter {encounter_id}")

        encounter_results = {encounter_id: {}}

        # Process all questions for this encounter
        for i, (encounter_id, base_qid) in enumerate(encounter_pairs):
            print(f"Processing question {i+1}/{len(encounter_pairs)}: {base_qid}")

            sample_data = agentic_data.get_combined_data(encounter_id, base_qid)
            if not sample_data:
                print(f"Warning: No data found for {encounter_id}, {base_qid}")
                continue

            try:
                # Process through the complete pipeline
                result = self.process_question(sample_data)
                
                encounter_results[encounter_id][base_qid] = {
                    "query_context": sample_data['query_context'],
                    "options": sample_data['options'],
                    "model_predictions": sample_data['model_predictions'],
                    "final_answer": result["final_answer"],
                    "confidence": result["confidence"],
                    "reasoning": result.get("reasoning", {}),
                    "retrieved_knowledge": result.get("retrieved_knowledge", {})
                }
                
            except Exception as e:
                print(f"Error processing {encounter_id}, {base_qid}: {e}")
                encounter_results[encounter_id][base_qid] = {
                    "query_context": sample_data['query_context'],
                    "options": sample_data['options'],
                    "model_predictions": sample_data['model_predictions'],
                    "final_answer": "Not mentioned",
                    "error": str(e)
                }

        # Save intermediate results
        output_file = os.path.join(
            self.args.output_dir if self.args else os.getcwd(), 
            f"diagnosis_based_rag_results_{encounter_id}.json"
        )
        
        with open(output_file, "w") as f:
            json.dump(encounter_results, f, indent=2)

        print(f"Processed all {len(encounter_pairs)} questions for encounter {encounter_id}")
        return encounter_results
    
    def format_results_for_evaluation(self, encounter_results, output_file):
        """Format results for official evaluation."""
        QIDS = [
            "CQID010-001",
            "CQID011-001", "CQID011-002", "CQID011-003", "CQID011-004", "CQID011-005", "CQID011-006",
            "CQID012-001", "CQID012-002", "CQID012-003", "CQID012-004", "CQID012-005", "CQID012-006",
            "CQID015-001",
            "CQID020-001", "CQID020-002", "CQID020-003", "CQID020-004", "CQID020-005", 
            "CQID020-006", "CQID020-007", "CQID020-008", "CQID020-009",
            "CQID025-001",
            "CQID034-001",
            "CQID035-001",
            "CQID036-001",
        ]
        
        qid_variants = {}
        for qid in QIDS:
            base_qid, variant = qid.split('-')
            if base_qid not in qid_variants:
                qid_variants[base_qid] = []
            qid_variants[base_qid].append(qid)
        
        required_base_qids = set(qid.split('-')[0] for qid in QIDS)
        
        formatted_predictions = []
        for encounter_id, questions in encounter_results.items():
            encounter_base_qids = set(questions.keys())
            if not required_base_qids.issubset(encounter_base_qids):
                print(f"Skipping encounter {encounter_id} - missing required questions")
                continue
            
            pred_entry = {'encounter_id': encounter_id}
            
            for base_qid, question_data in questions.items():
                if base_qid not in qid_variants:
                    continue
                
                final_answer = question_data['final_answer']
                options = question_data['options']
                
                not_mentioned_index = self._find_not_mentioned_index(options)
                
                self._process_answers(
                    pred_entry, 
                    base_qid, 
                    final_answer, 
                    options, 
                    qid_variants, 
                    not_mentioned_index
                )
            
            formatted_predictions.append(pred_entry)
        
        with open(output_file, 'w') as f:
            json.dump(formatted_predictions, f, indent=2)
        
        print(f"Formatted predictions saved to {output_file} ({len(formatted_predictions)} complete encounters)")
        return formatted_predictions
    
    def _find_not_mentioned_index(self, options):
        """Find the index of 'Not mentioned' in options."""
        for i, opt in enumerate(options):
            if opt.lower() == "not mentioned":
                return i
        return len(options) - 1
    
    def _process_answers(self, pred_entry, base_qid, final_answer, options, qid_variants, not_mentioned_index):
        """Process answers and add to prediction entry."""
        if ',' in final_answer:
            answer_parts = [part.strip() for part in final_answer.split(',')]
            answer_indices = []
            
            for part in answer_parts:
                found = False
                for i, opt in enumerate(options):
                    if part.lower() == opt.lower():
                        answer_indices.append(i)
                        found = True
                        break
                
                if not found:
                    answer_indices.append(not_mentioned_index)
            
            available_variants = qid_variants[base_qid]
            
            for i, idx in enumerate(answer_indices):
                if i < len(available_variants):
                    pred_entry[available_variants[i]] = idx
            
            for i in range(len(answer_indices), len(available_variants)):
                pred_entry[available_variants[i]] = not_mentioned_index
            
        else:
            answer_index = not_mentioned_index
            
            for i, opt in enumerate(options):
                if final_answer.lower() == opt.lower():
                    answer_index = i
                    break
            
            pred_entry[qid_variants[base_qid][0]] = answer_index
            
            if len(qid_variants[base_qid]) > 1:
                for i in range(1, len(qid_variants[base_qid])):
                    pred_entry[qid_variants[base_qid][i]] = not_mentioned_index


def run_diagnosis_based_pipeline_all_encounters(args=None):
    """Run the diagnosis-based pipeline for all available encounters."""
    if args is None:
        args = Args(use_finetuning=True, use_test_dataset=True)
        
    # Load data
    model_predictions_dict = DataLoader.load_all_model_predictions(args)
    if not model_predictions_dict:
        print("No model predictions found. Exiting.")
        return
        
    all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
    validation_df = DataLoader.load_validation_dataset(args)
    
    # Create agentic data manager
    agentic_data = AgenticRAGData(all_models_df, validation_df)
    
    # Initialize pipeline
    pipeline = AgenticDermatologyPipeline(args=args)
    
    # Get all unique encounters
    all_pairs = agentic_data.get_all_encounter_question_pairs()
    unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
    print(f"Found {len(unique_encounter_ids)} unique encounters to process")
    
    # Process all encounters
    all_encounter_results = {}
    
    for i, encounter_id in enumerate(unique_encounter_ids):
        print(f"\nProcessing encounter {i+1}/{len(unique_encounter_ids)}: {encounter_id}...")
        
        try:
            encounter_results = pipeline.process_single_encounter(agentic_data, encounter_id)
            if encounter_results:
                all_encounter_results.update(encounter_results)
                
            # Save intermediate results periodically
            if (i+1) % 5 == 0 or (i+1) == len(unique_encounter_ids):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                intermediate_file = os.path.join(
                    args.output_dir,
                    f"diagnosis_rag_intermediate_results_{timestamp}.json"
                )
                with open(intermediate_file, "w") as f:
                    json.dump(all_encounter_results, f, indent=2)
                print(f"Saved intermediate results to {intermediate_file}")
                
        except Exception as e:
            print(f"Error processing encounter {encounter_id}: {e}")
            traceback.print_exc()
            continue
    
    # Format and save final results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    formatted_output_file = os.path.join(
        args.output_dir,
        f"diagnosis_rag_formatted_predictions_{timestamp}.json"
    )
    
    formatted_predictions = pipeline.format_results_for_evaluation(
        all_encounter_results, 
        formatted_output_file
    )
    
    # Save complete results
    complete_output_file = os.path.join(
        args.output_dir,
        f"diagnosis_rag_complete_results_{timestamp}.json"
    )
    with open(complete_output_file, "w") as f:
        json.dump(all_encounter_results, f, indent=2)
    
    print(f"\nPipeline completed!")
    print(f"Complete results saved to: {complete_output_file}")
    print(f"Formatted predictions saved to: {formatted_output_file}")
    print(f"Total encounters processed: {len(all_encounter_results)}")
    print(f"Total complete encounters for evaluation: {len(formatted_predictions)}")
    
    return all_encounter_results, formatted_predictions


def run_diagnosis_based_pipeline_sample(args=None, num_samples=3):
    """Run the pipeline on a sample of encounters for testing."""
    if args is None:
        args = Args(use_finetuning=True, use_test_dataset=True)
    
    # Load data
    model_predictions_dict = DataLoader.load_all_model_predictions(args)
    if not model_predictions_dict:
        print("No model predictions found. Exiting.")
        return
        
    all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
    validation_df = DataLoader.load_validation_dataset(args)
    
    # Create agentic data manager
    agentic_data = AgenticRAGData(all_models_df, validation_df)
    
    # Initialize pipeline
    pipeline = AgenticDermatologyPipeline(args=args)
    
    # Get sample encounters
    all_pairs = agentic_data.get_all_encounter_question_pairs()
    unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
    
    # Sample random encounters
    sample_encounter_ids = random.sample(unique_encounter_ids, min(num_samples, len(unique_encounter_ids)))
    print(f"Testing on {len(sample_encounter_ids)} sample encounters: {sample_encounter_ids}")
    
    # Process sample encounters
    sample_results = {}
    
    for encounter_id in sample_encounter_ids:
        print(f"\nProcessing sample encounter: {encounter_id}")
        
        try:
            encounter_results = pipeline.process_single_encounter(agentic_data, encounter_id)
            if encounter_results:
                sample_results.update(encounter_results)
                
        except Exception as e:
            print(f"Error processing encounter {encounter_id}: {e}")
            traceback.print_exc()
    
    # Save sample results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sample_output_file = os.path.join(
        args.output_dir,
        f"diagnosis_rag_sample_results_{timestamp}.json"
    )
    
    with open(sample_output_file, "w") as f:
        json.dump(sample_results, f, indent=2)
    
    print(f"\nSample results saved to: {sample_output_file}")
    print(f"Processed {len(sample_results)} encounters")
    
    return sample_results


# Parameterizable Wrapper Classes
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for the RAG Pipeline."""
    
    # Model and dataset configuration
    use_finetuning: bool = True
    use_test_dataset: bool = True
    model_name: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    
    # Directory paths
    base_dir: Optional[str] = None
    output_dir: Optional[str] = None
    model_predictions_dir: Optional[str] = None
    images_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    
    # API configuration
    api_key: Optional[str] = None
    
    # Processing options
    max_reflection_cycles: int = 2
    confidence_threshold: float = 0.75
    save_intermediate_results: bool = True
    intermediate_save_frequency: int = 5

    # Gap 1 — Fast Triage Gatekeeper
    fast_triage_confidence_threshold: float = 0.95

    # Gap 2 — Asymmetric Partitioning
    enforce_modality_separation: bool = True
    
    # Knowledge base configuration
    knowledge_db_path: Optional[str] = None
    embedding_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    vector_dimension: int = 768
    top_k_semantic: int = 7
    top_k_keyword: int = 7
    top_k_hybrid: int = 10
    top_k_rerank: int = 5
    
    # Dataset configuration
    dataset_name_huggingface: str = "brucewayne0459/Skin_diseases_and_care"
    
    # Question type retrieval configuration
    question_type_retrieval_config: Optional[Dict[str, Dict[str, Any]]] = None
    default_rag_config: Optional[Dict[str, Any]] = None
    
    def to_rag_args(self) -> Args:
        """Convert to Args format."""
        # Set default configurations if not provided
        if self.question_type_retrieval_config is None:
            self.question_type_retrieval_config = {
                "Site Location": {"use_rag": False, "weight": 0.2},
                "Lesion Color": {"use_rag": False, "weight": 0.2},
                "Size": {"use_rag": False, "weight": 0.1},
                "Skin Description": {"use_rag": True, "weight": 0.3},
                "Onset": {"use_rag": True, "weight": 0.4},
                "Itch": {"use_rag": True, "weight": 0.4},
                "Extent": {"use_rag": False, "weight": 0.2},
                "Treatment": {"use_rag": True, "weight": 0.7},
                "Lesion Evolution": {"use_rag": True, "weight": 0.5},
                "Texture": {"use_rag": True, "weight": 0.3},
                "Lesion Count": {"use_rag": False, "weight": 0.1},
                "Differential": {"use_rag": True, "weight": 0.8},
                "Specific Diagnosis": {"use_rag": True, "weight": 0.8},
            }
        
        if self.default_rag_config is None:
            self.default_rag_config = {"use_rag": True, "weight": 0.4}
        
        # Pass all configuration parameters directly to Args constructor
        args = Args(
            use_finetuning=self.use_finetuning,
            use_test_dataset=self.use_test_dataset,
            base_dir=self.base_dir,
            output_dir=self.output_dir,
            model_predictions_dir=self.model_predictions_dir,
            images_dir=self.images_dir,
            dataset_path=self.dataset_path,
            gemini_model=self.gemini_model,
            max_reflection_cycles=self.max_reflection_cycles,
            confidence_threshold=self.confidence_threshold,
            knowledge_db_path=self.knowledge_db_path,
            embedding_model=self.embedding_model,
            cross_encoder_model=self.cross_encoder_model,
            vector_dimension=self.vector_dimension,
            top_k_semantic=self.top_k_semantic,
            top_k_keyword=self.top_k_keyword,
            top_k_hybrid=self.top_k_hybrid,
            top_k_rerank=self.top_k_rerank,
            dataset_name_huggingface=self.dataset_name_huggingface,
            question_type_retrieval_config=self.question_type_retrieval_config,
            default_rag_config=self.default_rag_config,
            fast_triage_confidence_threshold=self.fast_triage_confidence_threshold,
            enforce_modality_separation=self.enforce_modality_separation,
            model_name=self.model_name
        )
        
        return args


class RAGPipeline:
    """
    Main wrapper class for the diagnosis-based RAG medical analysis pipeline.
    
    This class provides a clean, parameterizable interface for running medical image
    analysis with knowledge retrieval, self-reflection, and multi-cycle reasoning.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or RAGConfig()
        self.args = self.config.to_rag_args()
        self.pipeline = None
        self.agentic_data = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize the pipeline components."""
        if self._initialized:
            return
            
        # Load environment variables for API key
        load_dotenv()
        
        # Initialize the main pipeline
        self.pipeline = AgenticDermatologyPipeline(
            api_key=self.config.api_key,
            args=self.args
        )
        # --- SURGICAL OVERRIDE START ---
        # Loads Qwen2-VL-2B-Instruct CSV predictions directly by glob-matching
        # the output directory, so the pipeline does not depend on DataLoader's
        # filename-timestamp parser.  Update MODEL_NAME below if you switch models.
        import pandas as pd
        import glob as _glob

        _MODEL_NAME   = "Qwen2-VL-2B-Instruct"
        _outputs_dir  = self.args.output_dir          # e.g. .../outputs/

        if self.args.use_test_dataset:
            _pattern = os.path.join(_outputs_dir, f"test_aggregated_predictions_{_MODEL_NAME}_*.csv")
            _label   = "test"
        else:
            _pattern = os.path.join(_outputs_dir, f"val_aggregated_predictions_{_MODEL_NAME}_*.csv")
            _label   = "validation"

        _matches = sorted(_glob.glob(_pattern))
        if not _matches:
            raise FileNotFoundError(
                f"[Surgical Override] No CSV found matching: {_pattern}\n"
                f"Run step4_validate.py first to generate the aggregated predictions file."
            )

        _csv_path = _matches[-1]   # pick the most-recently-written file
        print(f"[Surgical Override] Loading {_label} predictions from:\n  {_csv_path}")

        df = pd.read_csv(_csv_path)
        df["model_name"] = _MODEL_NAME          # inject the key DataLoader expects
        model_predictions_dict = {_label: df}
        print(f"[Surgical Override] Loaded {len(df)} rows for model '{_MODEL_NAME}'.")
        # --- SURGICAL OVERRIDE END ---
            
        all_models_df = self._concat_model_predictions(model_predictions_dict)
        dataset_df = DataLoader.load_validation_dataset(self.args)
        
        self.agentic_data = AgenticRAGData(all_models_df, dataset_df)
        self._initialized = True
        
    def _concat_model_predictions(self, model_predictions_dict: Dict) -> Any:
        """Safely concatenate model predictions."""
        if not model_predictions_dict:
            raise ValueError("No model predictions to concatenate")
            
        return pd.concat(model_predictions_dict.values(), ignore_index=True)
        
    def process_single_encounter(self, encounter_id: str) -> Dict[str, Any]:
        """
        Process a single encounter with RAG analysis.
        
        Args:
            encounter_id: The encounter ID to process
            
        Returns:
            Dictionary containing the processed results
        """
        if not self._initialized:
            self.initialize()
            
        encounter_results = self.pipeline.process_single_encounter(self.agentic_data, encounter_id)
        
        if not encounter_results:
            raise ValueError(f"No results generated for encounter {encounter_id}")
            
        return encounter_results
        
    def process_all_encounters(self, num_samples: Optional[int] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process all available encounters with RAG analysis.
        
        Args:
            num_samples: Optional limit on the number of encounters to process.
            
        Returns:
            Tuple of (complete_results, formatted_predictions)
        """
        if not self._initialized:
            self.initialize()
            
        all_pairs = self.agentic_data.get_all_encounter_question_pairs()
        unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
        
        if num_samples is not None:
            unique_encounter_ids = unique_encounter_ids[:num_samples]
            print(f"Limiting to first {num_samples} encounters for pilot validation.")
        
        print(f"Found {len(unique_encounter_ids)} unique encounters to process")
        
        all_encounter_results = {}
        
        for i, encounter_id in enumerate(unique_encounter_ids):
            print(f"Processing encounter {i+1}/{len(unique_encounter_ids)}: {encounter_id}...")
            
            try:
                encounter_results = self.pipeline.process_single_encounter(self.agentic_data, encounter_id)
                if encounter_results:
                    all_encounter_results.update(encounter_results)
                
                # Save intermediate results if configured
                if (self.config.save_intermediate_results and 
                    ((i+1) % self.config.intermediate_save_frequency == 0 or (i+1) == len(unique_encounter_ids))):
                    self._save_intermediate_results(all_encounter_results, i+1, len(unique_encounter_ids))
                    
            except Exception as e:
                print(f"Error processing encounter {encounter_id}: {e}")
                continue
        
        # Format and save final results
        return self._format_and_save_final_results(all_encounter_results, unique_encounter_ids)
    
    def process_sample_encounters(self, num_samples: int = 3) -> Dict[str, Any]:
        """
        Process a sample of encounters for testing.
        
        Args:
            num_samples: Number of encounters to sample
            
        Returns:
            Dictionary containing the processed sample results
        """
        if not self._initialized:
            self.initialize()
            
        all_pairs = self.agentic_data.get_all_encounter_question_pairs()
        unique_encounter_ids = sorted(list(set(pair[0] for pair in all_pairs)))
        
        # Sample random encounters
        sample_encounter_ids = random.sample(unique_encounter_ids, min(num_samples, len(unique_encounter_ids)))
        print(f"Processing {len(sample_encounter_ids)} sample encounters: {sample_encounter_ids}")
        
        sample_results = {}
        
        for encounter_id in sample_encounter_ids:
            print(f"Processing sample encounter: {encounter_id}")
            
            try:
                encounter_results = self.pipeline.process_single_encounter(self.agentic_data, encounter_id)
                if encounter_results:
                    sample_results.update(encounter_results)
                    
            except Exception as e:
                print(f"Error processing encounter {encounter_id}: {e}")
                continue
        
        # Save sample results
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        sample_output_file = os.path.join(
            self.args.output_dir,
            f"rag_sample_results_{timestamp}.json"
        )
        
        with open(sample_output_file, "w") as f:
            json.dump(sample_results, f, indent=2)
        
        print(f"Sample results saved to: {sample_output_file}")
        print(f"Processed {len(sample_results)} encounters")
        
        return sample_results
    
    def _save_intermediate_results(self, results: Dict, current: int, total: int) -> None:
        """Save intermediate results during processing."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        intermediate_output_file = os.path.join(
            self.args.output_dir, 
            f"rag_intermediate_results_{current}_of_{total}_{timestamp}.json"
        )
        with open(intermediate_output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved intermediate results after processing {current} encounters")
    
    def _format_and_save_final_results(self, all_encounter_results: Dict, unique_encounter_ids: List) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Format and save final results."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save complete results
        complete_output_file = os.path.join(
            self.args.output_dir, 
            f"{self.args.dataset_name}_data_cvqa_rag_complete_{timestamp}.json"
        )
        with open(complete_output_file, 'w') as f:
            json.dump(all_encounter_results, f, indent=2)
        
        # Format for evaluation
        formatted_output_file = os.path.join(
            self.args.output_dir, 
            f"{self.args.dataset_name}_data_cvqa_rag_formatted_{timestamp}.json"
        )
        
        formatted_predictions = self.pipeline.format_results_for_evaluation(all_encounter_results, formatted_output_file)
        
        print(f"Complete results saved to: {complete_output_file}")
        print(f"Formatted predictions saved to: {formatted_output_file}")
        print(f"Processed {len(formatted_predictions)} encounters successfully")
        
        return all_encounter_results, formatted_predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnosis-based RAG Medical Analysis Pipeline")
    parser.add_argument("--mode", choices=["all", "sample", "single"], default="sample",
                       help="Run mode: all encounters, sample, or single encounter")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to process in sample mode")
    parser.add_argument("--encounter_id", type=str,
                       help="Specific encounter ID to process in single mode")
    parser.add_argument("--use_test", action="store_true", default=True,
                       help="Use test dataset (default: True)")
    parser.add_argument("--use_finetuned", action="store_true", default=True,
                       help="Use fine-tuned model predictions (default: True)")
    
    cmd_args = parser.parse_args()
    
    # Initialize configuration
    args = Args(use_finetuning=cmd_args.use_finetuned, use_test_dataset=cmd_args.use_test)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load environment variables
    load_dotenv()
    
    if cmd_args.mode == "all":
        print("Running pipeline on all encounters...")
        run_diagnosis_based_pipeline_all_encounters(args)
        
    elif cmd_args.mode == "sample":
        print(f"Running pipeline on {cmd_args.num_samples} sample encounters...")
        run_diagnosis_based_pipeline_sample(args, num_samples=cmd_args.num_samples)
        
    elif cmd_args.mode == "single":
        if not cmd_args.encounter_id:
            print("Error: --encounter_id required for single mode")
            exit(1)
            
        print(f"Running pipeline on single encounter: {cmd_args.encounter_id}")
        
        # Load data
        model_predictions_dict = DataLoader.load_all_model_predictions(args)
        if not model_predictions_dict:
            print("No model predictions found. Exiting.")
            exit(1)
            
        all_models_df = pd.concat(model_predictions_dict.values(), ignore_index=True)
        validation_df = DataLoader.load_validation_dataset(args)
        
        # Create agentic data manager
        agentic_data = AgenticRAGData(all_models_df, validation_df)
        
        # Initialize pipeline
        pipeline = AgenticDermatologyPipeline(args=args)
        
        # Process single encounter
        results = pipeline.process_single_encounter(agentic_data, cmd_args.encounter_id)
        
        if results:
            # Format and save results
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                args.output_dir,
                f"diagnosis_rag_single_{cmd_args.encounter_id}_{timestamp}.json"
            )
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
                
            print(f"Results saved to: {output_file}")
        else:
            print(f"No results for encounter {cmd_args.encounter_id}")