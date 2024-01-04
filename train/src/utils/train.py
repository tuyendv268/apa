import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

import pickle
import json
import re
import os

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

def load_id(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

        lines = [line.strip() for line in lines]

    return lines

def load_data(data_dir):
    ids = load_id(f'{data_dir}/id')
    phone_ids = np.load(f'{data_dir}/phone_ids.npy')
    word_ids = np.load(f'{data_dir}/word_ids.npy')
    
    phone_scores = np.load(f'{data_dir}/phone_scores.npy')
    word_scores = np.load(f'{data_dir}/word_scores.npy')
    sentence_scores = np.load(f'{data_dir}/sentence_scores.npy')

    durations = np.load(f'{data_dir}/duration.npy')
    gops = np.load(f'{data_dir}/gop.npy')
    wavlm_features_path = f'{data_dir}/wavlm_features'

    relative_positions = np.load(f'{data_dir}/relative_positions.npy')

    return ids, phone_ids, word_ids, phone_scores, word_scores, \
        sentence_scores, durations, gops, relative_positions, wavlm_features_path

def to_device(batch, device):
    ids = batch["ids"]
    features = batch["features"].to(device)
    phone_ids = batch["phone_ids"].to(device)
    relative_positions = batch["relative_positions"].to(device)
    word_ids = batch["word_ids"]
    
    phone_labels = batch["phone_scores"].to(device)
    word_labels = batch["word_scores"].to(device)
    utterance_labels = batch["sentence_scores"].to(device)

    return ids, features, phone_ids, word_ids, relative_positions, \
        phone_labels, word_labels, utterance_labels

def convert_score_to_color(score, YELLOW_GREEN=80/50, RED_YELLOW=30/50):
    if RED_YELLOW is not None:
        LABEL2ID = {"GREEN": 0, "YELLOW": 1, "RED":2}
        red_index = score < RED_YELLOW
        yellow_index = ((score >= RED_YELLOW).int() & (score < YELLOW_GREEN).int()).bool()
        green_index = score >= YELLOW_GREEN
    else:
        LABEL2ID = {"GREEN": 0, "YELLOW": 1, "RED":1}
        RED_YELLOW = 30/50
        red_index = score < RED_YELLOW
        yellow_index = ((score >= RED_YELLOW).int() & (score < YELLOW_GREEN).int()).bool()
        green_index = score >= YELLOW_GREEN


    score[red_index] = LABEL2ID["RED"]
    score[yellow_index] = LABEL2ID["YELLOW"]
    score[green_index] = LABEL2ID["GREEN"]

    return score

def valid_phn(predict, target):
    preds, targs = [], []

    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if target[i, j] >= 0:
                preds.append(predict[i, j])
                targs.append(target[i, j])
    targs = np.array(targs)
    preds = np.array(preds)

    mse = np.mean((targs - preds) ** 2)
    mae = np.mean(np.abs(targs - preds))
    corr = np.corrcoef(preds, targs)[0, 1]
    return mse, mae, corr

def valid_wrd(predict, target, word_id):
    preds, targs = [], []

    for i in range(target.shape[0]):
        prev_w_id, start_id = 0, 0
        # for each token
        for j in range(target.shape[1]):
            cur_w_id = word_id[i, j].int()
            # if a new word
            if cur_w_id != prev_w_id:
                # average each phone belongs to the word
                preds.append(np.mean(predict[i, start_id: j].numpy(), axis=0))
                targs.append(np.mean(target[i, start_id: j].numpy(), axis=0))

                if cur_w_id == -1:
                    break
                else:
                    prev_w_id = cur_w_id
                    start_id = j

    preds = np.array(preds)
    targs = np.array(targs).round(2)

    word_mse = np.mean((preds - targs) ** 2)
    wrd_mae = np.mean(np.abs(preds - targs))
    word_corr = np.corrcoef(preds, targs)[0, 1]
    
    return word_mse, wrd_mae, word_corr

def valid_utt(predict, target):
    utt_mse = np.mean(((predict[:, 0] - target[:, 0]) ** 2).numpy())
    utt_mae = np.mean((np.abs(predict[:, 0] - target[:, 0])).numpy())
    
    utt_corr = np.corrcoef(predict[:, 0], target[:, 0])[0, 1]
    return utt_mse, utt_mae, utt_corr

def to_cpu(preds, labels):
    preds = preds.detach().cpu().squeeze(-1)
    labels = labels.detach().cpu()

    return preds, labels

def load_pred_and_label(pred_path, label_path):
    pred = np.load(pred_path)
    label = np.load(label_path)

    pred = np.concatenate(pred)
    label = np.concatenate(label)
    index = label != -1    
    
    return label[index], pred[index]

def save_confusion_matrix_figure(
        fig_path, pred_path, label_path, YELLOW_GREEN=80/50, RED_YELLOW=30/50):
    
    label, pred = load_pred_and_label(
        pred_path=pred_path, label_path=label_path)
    
    actual = convert_score_to_color(
        torch.from_numpy(label), 
        YELLOW_GREEN=YELLOW_GREEN, 
        RED_YELLOW=RED_YELLOW
    )
    
    predicted = convert_score_to_color(
        torch.from_numpy(pred), 
        YELLOW_GREEN=YELLOW_GREEN, 
        RED_YELLOW=RED_YELLOW
    )
    
    cfs_mtr = confusion_matrix(actual, predicted)
    cfs_mtr = cfs_mtr / cfs_mtr.sum(axis=1, keepdims=True)
    if RED_YELLOW is not None:
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix = cfs_mtr, 
            # display_labels = ["GREEN", "YELLOW", "RED"]
        )
    else:
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix = cfs_mtr, 
            # display_labels = ["CORRECT", "INCORRECT"]
        )

    plt.title("Confusion Matrix")
    cm_display.plot(cmap='Blues')
    plt.savefig(fig_path) 
    plt.close()

def save(epoch, output_dir, model, optimizer, phone_desicion_result, \
    phone_predicts, phone_labels, word_predicts, word_labels, utterance_predicts, utterance_labels):
    
    model_path = f'{output_dir}/model.pt'
    optimizer_path = f'{output_dir}/optimizer.pt'
    phone_desicion_result_path = f'{output_dir}/phone_result'

    phone_predict_path = f'{output_dir}/phn_pred.npy'
    phone_label_path = f'{output_dir}/phn_label.npy'
    word_predict_path = f'{output_dir}/wrd_pred.npy'
    word_label_path = f'{output_dir}/wrd_label.npy'
    utterance_predict_path = f'{output_dir}/utt_pred.npy'
    utterance_label_path = f'{output_dir}/utt_label.npy'

    three_class_fig_path = f'{output_dir}/confusion_matrix_three_class.png'
    two_class_fig_path = f'{output_dir}/confusion_matrix_two_class.png'

    with open(phone_desicion_result_path, "w") as f:
        f.write(phone_desicion_result)

    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    np.save(phone_predict_path, phone_predicts)
    np.save(phone_label_path, phone_labels)
    np.save(word_predict_path, word_predicts)
    np.save(word_label_path, word_labels)
    np.save(utterance_predict_path, utterance_predicts)
    np.save(utterance_label_path, utterance_labels)

    save_confusion_matrix_figure(
        three_class_fig_path, phone_predict_path, phone_label_path, 
        YELLOW_GREEN=80/50, RED_YELLOW=40/50
    )
    
    save_confusion_matrix_figure(
        two_class_fig_path, phone_predict_path, phone_label_path, 
        YELLOW_GREEN=80/50, RED_YELLOW=None
    )

    print(f'Save state dict and result to {output_dir}')

@torch.no_grad()
def validate(epoch, gopt_model, optimizer, testloader, best_mse, ckpt_dir, device="cuda"):
    gopt_model.eval()
    A_phn, A_phn_target = [], []
    A_utt, A_utt_target = [], []
    A_wrd, A_wrd_target, A_wrd_id = [], [], []

    for batch in testloader:
        ids, features, phone_ids, word_ids, relative_positions,\
            phone_labels, word_labels, utterance_labels = to_device(batch, device)
        
        utterance_preds, phone_preds, word_preds = gopt_model(
            x=features.float(), phn=phone_ids.long(), rel_pos=relative_positions.long())
        
        phone_preds, phone_labels = to_cpu(phone_preds, phone_labels)
        word_preds, word_labels = to_cpu(word_preds, word_labels)
        utterance_preds, utterance_labels = to_cpu(utterance_preds, utterance_labels)
        
        A_phn.append(phone_preds), A_phn_target.append(phone_labels)
        A_utt.append(utterance_preds), A_utt_target.append(utterance_labels)
        A_wrd.append(word_preds), A_wrd_target.append(word_labels), A_wrd_id.append(word_ids)
    
    # phone level
    A_phn, A_phn_target  = torch.vstack(A_phn), torch.vstack(A_phn_target)
    decision_result = calculate_phone_decision_result(A_phn, A_phn_target)

    # word level
    A_word, A_word_target, A_word_id = \
        torch.vstack(A_wrd), torch.vstack(A_wrd_target), torch.vstack(A_wrd_id) 

    # utterance level
    A_utt, A_utt_target = torch.vstack(A_utt), torch.vstack(A_utt_target)

    # valid_token_mse, mae, corr
    phn_mse, phn_mae, phn_corr = valid_phn(A_phn, A_phn_target)
    word_mse, wrd_mae, word_corr = valid_wrd(A_word, A_word_target, A_word_id)
    utt_mse, utt_mae, utt_corr = valid_utt(A_utt, A_utt_target)

    if phn_mse < best_mse:
        best_mse = phn_mse
    ckpt_dir = f'{ckpt_dir}/ckpts-eph={epoch}-mse={round(phn_mse, 4)}'
    os.makedirs(ckpt_dir)
    
    save(
        epoch=epoch,
        output_dir=ckpt_dir, 
        model=gopt_model, 
        optimizer=optimizer, 
        phone_desicion_result=decision_result,
        phone_predicts=A_phn.numpy(), 
        phone_labels=A_phn_target.numpy(), 
        word_predicts=A_word.numpy(), 
        word_labels=A_word_target.numpy(), 
        utterance_predicts=A_utt.numpy(), 
        utterance_labels=A_utt_target.numpy()
    )
    
    with open(f'{ckpt_dir}/pcc', "w") as f:
        f.write("Phone level:  MSE={:.3f}  MAE={:.3f}  PCC={:.3f} \n".format(phn_mse, phn_mae, phn_corr))
        f.write("Word level:  MSE={:.3f}  MAE={:.3f}  PCC={:.3f} \n".format(word_mse, wrd_mae, word_corr))
        f.write("Utt level:  MSE={:.3f}  MAE={:.3f}  PCC={:.3f} \n".format(utt_mse, utt_mae, utt_corr))

    print(f"### Validation result (epoch={epoch})")
    print("  Phone level:  MSE={:.3f}  MAE={:.3f}  PCC={:.3f} ".format(phn_mse, phn_mae, phn_corr))
    print("   Word level:  MSE={:.3f}  MAE={:.3f}  PCC={:.3f} ".format(word_mse, wrd_mae, word_corr))
    print("    Utt level:  MSE={:.3f}  MAE={:.3f}  PCC={:.3f} ".format(utt_mse, utt_mae, utt_corr))

    return {
        "phn_mse": phn_mse, 
        "phn_mae": phn_mae,
        "phn_corr": phn_corr,
        "word_mse": word_mse,
        "wrd_mae": wrd_mae,
        "word_corr": word_corr,
        "utt_mse": utt_mse,
        "utt_mae": utt_mae,
        "utt_corr": utt_corr,
        "best_mse": best_mse
    }

def calculate_phone_decision_result(A_phn, A_phn_target):
    indices = A_phn_target != -1
    _label = A_phn_target[indices].clone()
    _pred = A_phn[indices].clone()

    converted_pred = convert_score_to_color(_pred).view(-1)
    converted_label = convert_score_to_color(_label).view(-1)

    result = classification_report(y_true=converted_label, y_pred=converted_pred)
    print("### F1 Score: \n", result)

    return result

