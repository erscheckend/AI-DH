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
        
        # Initialize lists to store the scores
        bleu_scores = []
        rouge_scores = []
        chrf_scores = []
        meteor_scores = []

        # Loop through each row to calculate and collect scores
        for row_num, row in enumerate(reader, start=2):
            reference = row[3]
            hypotheses = {
                "Column E": row[4],
                "Column F": row[5],
            }
            
            # Collect BLEU scores
            bleu_row = {}
            for label, hypothesis in hypotheses.items():
                bleu_row[label] = calculate_bleu(reference, hypothesis)
            bleu_scores.append(bleu_row)
            
            # Collect ROUGE-L scores
            rouge_row = {}
            for label, hypothesis in hypotheses.items():
                rouge_row[label] = calculate_rouge_l(reference, hypothesis)
            rouge_scores.append(rouge_row)
            
            # Collect chrF scores
            chrf_row = {}
            for label, hypothesis in hypotheses.items():
                chrf_row[label] = calculate_chrf(reference, hypothesis)
            chrf_scores.append(chrf_row)
            
            # Collect METEOR scores
            meteor_row = {}
            for label, hypothesis in hypotheses.items():
                meteor_row[label] = calculate_meteor(reference, hypothesis)
            meteor_scores.append(meteor_row)

        # Print all BLEU scores
        print("BLEU Scores:")
        for i, score in enumerate(bleu_scores):
            print(f"Row {i+2} - Gemini: {score['Column E']:.2f}, Yandex: {score['Column F']:.2f}")
        print("\n" + "-"*50 + "\n")

        # Print all ROUGE scores
        print("ROUGE Scores:")
        for i, score in enumerate(rouge_scores):
            print(f"Row {i+2} - Gemini: {score['Column E']:.2f}, Yandex: {score['Column F']:.2f}")
        print("\n" + "-"*50 + "\n")

        # Print all chrF scores
        print("chrF Scores:")
        for i, score in enumerate(chrf_scores):
            print(f"Row {i+2} - Gemini: {score['Column E']:.2f}, Yandex: {score['Column F']:.2f}")
        print("\n" + "-"*50 + "\n")

        # Print all METEOR scores
        print("METEOR Scores:")
        for i, score in enumerate(meteor_scores):
            print(f"Row {i+2} - Gemini: {score['Column E']:.2f}, Yandex: {score['Column F']:.2f}")
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
