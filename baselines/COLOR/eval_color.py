import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


def evaluate(model, loader):
    model.eval()
    similarities = []
    with torch.no_grad():
        for inputs in tqdm(loader):
            user_utt_ids, user_utt_mask = inputs["user_utterance"]
            delta_follow_ids, delta_follow_mask = inputs["delta_follow"]
            start_subgoal_ids, start_subgoal_mask = inputs["start_subgoal"]
            target_subgoal_ids, target_subgoal_mask = inputs["target_subgoal"]
            gold_bridge_list = inputs["gold_bridge_list"][0]  # note that bzs=1
            if len(gold_bridge_list) < 1:
                continue

            start_latent = model.get_time_control_embed(start_subgoal_ids, start_subgoal_mask)  # [1, 768]
            target_latent = model.get_time_control_embed(target_subgoal_ids, target_subgoal_mask)  # [1, 768]
            Z_u = model.get_user_utt_representation(user_utt_ids, user_utt_mask)  # [1, 768]
            delta_u = model.get_delta_u_representation(delta_follow_ids, delta_follow_mask)  # [1]

            simulate_bridge_points = model.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent,
                                                                    T=len(gold_bridge_list) + 1, Z_u=Z_u,
                                                                    delta_u=delta_u)
            gold_bridge_list = [start_latent] + gold_bridge_list
            assert len(simulate_bridge_points) == len(gold_bridge_list)

            for i in range(len(simulate_bridge_points)):
                similarity = F.cosine_similarity(simulate_bridge_points[i], gold_bridge_list[i], dim=-1).item()
                similarities.append(similarity)
    avg_similarity = np.mean(similarities)
    result = {
        "avg_similarity": avg_similarity
    }
    return result


def evaluate_brownian_bridge(model, loader, output_path):
    model.eval()
    similarities = []
    with open(output_path, 'w', encoding='utf-8') as fw:
        with torch.no_grad():
            for inputs in tqdm(loader):
                similarity_list = []
                user_utt_ids, user_utt_mask = inputs["user_utterance"]
                delta_follow_ids, delta_follow_mask = inputs["delta_follow"]
                start_subgoal_ids, start_subgoal_mask = inputs["start_subgoal"]
                target_subgoal_ids, target_subgoal_mask = inputs["target_subgoal"]
                gold_bridge_list = inputs["gold_bridge_list"][0]  # note that bzs=1
                if len(gold_bridge_list) < 1:
                    continue

                start_latent = model.get_time_control_embed(start_subgoal_ids, start_subgoal_mask)  # [1, 768]
                target_latent = model.get_time_control_embed(target_subgoal_ids, target_subgoal_mask)  # [1, 768]
                Z_u = model.get_user_utt_representation(user_utt_ids, user_utt_mask)  # [1, 768]
                delta_u = model.get_delta_u_representation(delta_follow_ids, delta_follow_mask)  # [1]

                simulate_bridge_points = model.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent,
                                                                        T=len(gold_bridge_list) + 1, Z_u=Z_u,
                                                                        delta_u=delta_u)
                gold_bridge_list = [start_latent] + gold_bridge_list
                assert len(simulate_bridge_points) == len(gold_bridge_list)

                for i in range(len(simulate_bridge_points)):
                    similarity = F.cosine_similarity(simulate_bridge_points[i], gold_bridge_list[i], dim=-1).item()
                    similarity_list.append(similarity)
                    similarities.append(similarity)
                fw.write(" ".join([str(sim) for sim in similarity_list]) + "\n")
    avg_similarity = np.mean(similarities)
    return avg_similarity


def evaluate_planning(args, model, loader):
    lm_accs, lm_losss, trans_accs, trans_losss, total_losss = [], [], [], [], []
    with torch.no_grad():
        for inputs in tqdm(loader):
            input_ids, input_masks = inputs["input"]
            decoder_input_all_ids, decoder_input_all_masks = inputs["decoder_input_all"]
            labels, labels_mask = inputs["label"]
            transition_number_label = inputs["transition_number"]
            simulate_bridge_embed, simulate_bridge_mask = inputs["simulate_bridge_embed"]
            gold_bridge_embed, gold_bridge_mask = inputs["gold_bridge_embed"]

            if args.use_simulated:
                model_output = model(input_ids=input_ids, attention_mask=input_masks,
                                     decoder_input_ids=decoder_input_all_ids,
                                     decoder_attention_mask=decoder_input_all_masks, labels=labels,
                                     bridge_embeds=simulate_bridge_embed, bridge_mask=simulate_bridge_mask,
                                     transition_number_label=transition_number_label)
            else:
                model_output = model(input_ids=input_ids, attention_mask=input_masks,
                                     decoder_input_ids=decoder_input_all_ids,
                                     decoder_attention_mask=decoder_input_all_masks, labels=labels,
                                     bridge_embeds=gold_bridge_embed, bridge_mask=gold_bridge_mask,
                                     transition_number_label=transition_number_label)

            lm_logits = model_output["lm_logits"]
            lm_loss = model_output["lm_loss"]
            if args.train_use_bridge:
                trans_logits = model_output["trans_logits"]
                trans_loss = model_output["trans_loss"]
                trans_acc = compute_trans_acc(trans_logits, transition_number_label)
                if args.use_KLD:
                    kl_loss = model_output["kl_loss"]
                    total_loss = args.trans_alpha * trans_loss + args.gen_beta * lm_loss + args.kl_gamma * kl_loss
                else:
                    total_loss = args.trans_alpha * trans_loss + args.gen_beta * lm_loss
            else:
                trans_loss = 0.0
                trans_acc = 0.0
                kl_loss = 0.0
                total_loss = lm_loss
            lm_acc = compute_acc(lm_logits, labels_mask, labels)
            lm_accs.append(lm_acc)
            trans_accs.append(trans_acc)

            if args.gradient_accumulation_steps > 0:
                lm_losss.append(float(lm_loss) / args.gradient_accumulation_steps)
                trans_losss.append(float(trans_loss) / args.gradient_accumulation_steps)
                total_losss.append(float(total_loss) / args.gradient_accumulation_steps)
            else:
                lm_losss.append(float(lm_loss))
                trans_losss.append(float(trans_loss))
                total_losss.append(float(total_loss))
    avg_lm_acc = np.mean(lm_accs)
    avg_lm_loss = np.mean(lm_losss)
    avg_trans_acc = np.mean(trans_accs)
    avg_trans_loss = np.mean(trans_losss)
    avg_total_loss = np.mean(total_losss)

    return_dict = {
        "avg_lm_acc": avg_lm_acc,
        "avg_lm_loss": avg_lm_loss,
        "avg_trans_acc": avg_trans_acc,
        "avg_trans_loss": avg_trans_loss,
        "avg_total_loss": avg_total_loss,
    }
    return return_dict


def compute_acc(lm_logits, seq_masks, seq_labels):
    pred = torch.softmax(lm_logits, -1)
    _, pred_y = pred.max(-1)
    hit_tokens = (torch.eq(pred_y, seq_labels).float() * seq_masks).sum().item()
    num_tokens = seq_masks.float().sum().item()
    acc = float(hit_tokens) / num_tokens if num_tokens > 0 else 0.0
    return acc


def compute_trans_acc(transition_logits, transition_labels):
    pred = torch.softmax(transition_logits, -1)
    _, pred_y = pred.max(-1)
    hit_tokens = (torch.eq(pred_y, transition_labels).float()).sum().item()
    num_tokens = transition_labels.size(0)
    acc = float(hit_tokens) / num_tokens if num_tokens > 0 else 0.0
    return acc
