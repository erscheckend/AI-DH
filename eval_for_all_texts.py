import csv
import sys
from sacrebleu import sentence_bleu
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

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

# Unified function to process the CSV, calculate all metrics, and save results
def process_csv(file_path):
    output_rows = []  # List to collect output rows

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        headers = next(reader)
        
        # Append new headers for all metrics to the existing headers
        output_rows.append(headers + [
            "BLEU score GPT", "ROUGE score GPT", "chrF scoreGPT", "METEOR scoreGPT",
            "BLEU score Google", "ROUGE score Google", "chrF score Google", "METEOR score Google",
            "BLEU score Gemini", "ROUGE score Gemini", "chrF score Gemini", "METEOR score Gemini",
            "BLEU score Yandex", "ROUGE score Yandex", "chrF score Yandex", "METEOR score Yandex"
        ])
        
        for row_num, row in enumerate(reader, start=2):
            # Extract reference and hypotheses from specific columns
            reference = row[2]  # Column C
            hypothesis_gpt = row[3]  # Column D
            hypothesis_google = row[4]  # Column E
            hypothesis_gemini = row[5]  # Column F
            hypothesis_yandex = row[6]  # Column G

            # Calculate each metric for all hypotheses
            scores = []
            for hypothesis in [hypothesis_gpt, hypothesis_google, hypothesis_gemini, hypothesis_yandex]:
                bleu = round(calculate_bleu(reference, hypothesis), 2)
                rouge = round(calculate_rouge_l(reference, hypothesis), 2)
                chrf = round(calculate_chrf(reference, hypothesis), 2)
                meteor = round(calculate_meteor(reference, hypothesis), 2)
                scores.extend([bleu, rouge, chrf, meteor])

            # Append calculated scores to the row
            output_rows.append(row + scores)

    # Write the output to a new CSV file with "_with_scores" suffix
    output_file_path = file_path.replace('.csv', '_with_scores.csv')
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(output_rows)

    print(f"Evaluation results saved to {output_file_path}")

# Main entry point
def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    process_csv(file_path)

if __name__ == "__main__":
    main()
