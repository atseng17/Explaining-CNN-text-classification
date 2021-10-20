import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import pickle

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):# there are 256 epoches. The dataset had 156 batchs of size 64

        for batch in train_iter:

            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            print('logit vector',logit, logit.size())
            print('target vector',target, target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1 # this is the number of batches
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                print(loss)
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss, 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        # print(loss)
        # print(logit)
        # print(torch.max(logit, 1))

        avg_loss += loss
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()) == target).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

class PGD():
    def __init__(self, model,attacks_lst_PGDM):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.attacks_lst_PGDM = attacks_lst_PGDM

    def attack(self, epsilon=1.5, alpha=0.3, emb_name='embed', is_first_attack=False):
        # emb_name='embed'< this is the name of the embedding layer in this model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embed'):
        # emb_name='embed'< this is the name of the embedding layer in this model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        self.attacks_lst_PGDM.append(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]



def predict_and_attack(text, model, text_field, label_feild, cuda_flag, exp_save_path):
    assert isinstance(text, str)
    model.eval()

    attacks_lst_PGDM = []
    pred_adv_lst_ = []
    pgd = PGD(model,attacks_lst_PGDM)
    K = 10

    text = text_field.preprocess(text)

    np.save(os.path.join(exp_save_path,'text.npy'), text)

    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)

    # print('tokenized input:',x)
    if cuda_flag:
        x = x.cuda()

    # The First forward path
    output = model(x)# print("output:",output)
    _, predicted = torch.max(output, 1)# 

    if cuda_flag:
        target = predicted.cuda()
    else:
        target = predicted
    loss = F.cross_entropy(output, target, reduction='sum')
    loss.backward()

    pgd.backup_grad()
    for t in range(K):
        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != K-1:
            model.zero_grad()
        else:
            pgd.restore_grad()
        # The Kth forward paths
        output_adv = model(x)
        # print('output_adv:',output_adv)
        _, predicted_adv = torch.max(output_adv, 1) # print('attacked pred:',predicted_adv)
        pred_adv_lst_.append(predicted_adv)
        loss_adv = F.cross_entropy(output_adv, target, reduction='sum')
        loss_adv.backward()

    # The Second forward path
    output_adv = model(x)
    _, predicted_adv = torch.max(output_adv, 1)
    # print('final attacked output:', label_feild.vocab.itos[predicted_adv[0]+1])

    # save results
    attacks_lst_PGDM_=[]
    for att in attacks_lst_PGDM:
        attacks_lst_PGDM_.append(np.array(att))

    np.save(os.path.join(exp_save_path,'x_index.npy'), np.array(x[0].cpu()))

    pred_adv_lst__=[]
    for predlabels in pred_adv_lst_:
        pred_adv_lst__.append(predlabels.numpy())

    np.save(os.path.join(exp_save_path,'pred_adv_lst.npy'), np.array(pred_adv_lst__))
    np.save(os.path.join(exp_save_path,'attacks_lst_PGDM.npy'), np.array(attacks_lst_PGDM_))

    return label_feild.vocab.itos[predicted[0]+1]  


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
    with open('saved_data/attacks_lst_PGDM_K10_000.pickle', "wb") as output_file:
        pickle.dump(total_list_att, output_file)

