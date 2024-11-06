import csv
import sys
from sacrebleu import sentence_bleu
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK resources for METEOR scoring
nltk.download('punkt')

# Define individual score calculation functions
def calculate_bleu(reference, hypothesis):
    return sentence_bleu(hypothesis, [reference]).score

def calculate_rouge_l(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure * 100

def calculate_chrf(reference, hypothesis):
    chrf_scorer = CHRF()
    score = chrf_scorer.sentence_score(hypothesis, [reference])
    return score.score

def calculate_meteor(reference, hypothesis):
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)
    return meteor_score([reference_tokens], hypothesis_tokens) * 100

# Unified function to process the CSV and calculate all metrics
def process_csv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        
        for row_num, row in enumerate(reader, start=2):
            reference = row[3]
            hypotheses = {
                "Column E": row[4],
                "Column F": row[5],
            }
            
            # Print BLEU scores
            print("BLEU scores")
            for label, hypothesis in hypotheses.items():
                score = calculate_bleu(reference, hypothesis)
                print(f"Row {row_num} - {label} score: {score:.2f}")
            print()
            
            # Print ROUGE-L scores
            print("ROUGE scores")
            for label, hypothesis in hypotheses.items():
                score = calculate_rouge_l(reference, hypothesis)
                print(f"Row {row_num} - {label} score: {score:.2f}")
            print()
            
            # Print chrF scores
            print("chrF scores")
            for label, hypothesis in hypotheses.items():
                score = calculate_chrf(reference, hypothesis)
                print(f"Row {row_num} - {label} score: {score:.2f}")
            print()
            
            # Print METEOR scores
            print("METEOR scores")
            for label, hypothesis in hypotheses.items():
                score = calculate_meteor(reference, hypothesis)
                print(f"Row {row_num} - {label} score: {score:.2f}")
            print("\n" + "-"*50 + "\n")

# Main entry point
def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    process_csv(file_path)

if __name__ == "__main__":
    main()
