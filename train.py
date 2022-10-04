import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os, random, cv2, argparse
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import audio
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data as data_utils
from hparams import hparams
from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual, Wav2Lip_decoder

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True)
parser.add_argument('--checkpoint', help='Save checkpoints to this directory', required=True)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained audio-visual sync module', required=True)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))


syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.split = split
        filelist = glob(os.path.join(args.data_root, f'*/{self.split}/*'))
        self.all_videos = filelist
        print(self.split, len(filelist))

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)
        if start_id - 2 < 0: return None
        window_fnames = []
        for frame_id in range(start_id - 2, start_id + syncnet_T + 2):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None
            window.append(img)
        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)
        mels = np.asarray(mels)
        return mels

    def prepare_window(self, window):
        x = np.asarray(window) / (255. / 2) - 1
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            img_id = self.get_frame_id(img_name)
            wrong_id = self.get_frame_id(wrong_img_name)
            if img_id - 2 < 0 or wrong_id - 2 < 0 or img_id + 3 > len(img_names) or wrong_id + 3 > len(img_names):
                continue

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames[2:7])
            if wrong_window is None:
                continue

            try:
                wavpath = vidname.replace('LRW-wav2lip', 'LRW-audio') + '.wav'
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            lip_y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window[:, 2:7], wrong_window], axis=0)

            lip_y[:, :, :window.shape[2] // 2] = 0.
            lip = []
            for l in range(2, lip_y.shape[1] - 2):
                lip.append(lip_y[:, l - 2:l + 3])

            lip = np.stack(lip, axis=0)

            x = torch.FloatTensor(x)  # B,6,5,128,128
            mel = torch.FloatTensor(mel.T).unsqueeze(0)  # B,1,80,16
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)  # B,5,1,80,16
            y = torch.FloatTensor(y[:, 2:7])  # B,3,5,128,128
            lip = torch.FloatTensor(lip)  # B,5,3,5,128,128
            return x, indiv_mels, mel, y, lip


recon_loss = nn.L1Loss()
logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

def get_sync_loss(mel, g, syncnet):
    g = g[:, :, :, g.size(3) // 2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def train(device, model, syncnet, disc, train_data_loader, test_data_loader, optimizer,
          disc_optimizer,
          checkpoint_dir=None, nepochs=None):
    global_epoch = 0
    global_step = 0
    print(f'training for {len(train_data_loader)} steps')
    writer = SummaryWriter(log_dir=f'runs/{(args.checkpoint_dir).split("/")[-1]}')
    model, dec = model

    best_l1_loss, best_sync_loss = 100000, 100000
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss, running_disc_real_loss, running_disc_fake_loss = 0., 0., 0., 0., 0., 0.
        running_reconst_loss, running_add_loss, running_lip_loss = 0., 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt, lip) in prog_bar:
            disc.train()
            model.train()
            dec.train()
            syncnet.eval()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            lip = lip.to(device)

            ### Train generator
            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            te_fusion, tr_fusion, feats, mem_recon, add_loss, key_add, value_add, lip_embed = model(indiv_mels, x, lip,False)
            g_te = dec(te_fusion, feats)
            g_tr = dec(tr_fusion, feats)

            g_te_lip = g_te.clone().unsqueeze(1)
            g_tr_lip = g_tr.clone().unsqueeze(1)
            g_te_lip[:, :, :, :, :g_te_lip.shape[4]//2:, :] = 0.
            g_tr_lip[:, :, :, :, :g_tr_lip.shape[4]//2:, :] = 0.
            g_te_lip_embed, g_tr_lip_embed = model.module.lip_check(g_te_lip, g_tr_lip)
            lip_loss_te = recon_loss(lip_embed, g_te_lip_embed)
            lip_loss_tr = recon_loss(lip_embed, g_tr_lip_embed)
            lip_loss_tot = (lip_loss_te + lip_loss_tr)

            sync_loss = get_sync_loss(mel, (g_te + 1) / 2, syncnet)
            sync_loss_o = get_sync_loss(mel, (g_tr + 1) / 2, syncnet)
            sync_loss_tot = (sync_loss + sync_loss_o)

            g_te_perceptual = disc.forward(g_te)
            g_te_perceptual = F.binary_cross_entropy(g_te_perceptual, torch.ones((len(g_te_perceptual), 1)).cuda())
            g_tr_perceptual = disc.forward(g_tr)
            g_tr_perceptual = F.binary_cross_entropy(g_tr_perceptual, torch.ones((len(g_tr_perceptual), 1)).cuda())
            perceptual_loss = g_te_perceptual + g_tr_perceptual

            l1loss = recon_loss((g_te + 1) / 2, (gt + 1) / 2)
            l1loss_o = recon_loss((g_tr + 1) / 2, (gt + 1) / 2)
            l1loss_tot = l1loss + l1loss_o

            loss = hparams.l1_wt * l1loss_tot + hparams.av_wt * sync_loss_tot +  hparams.vv_wt * lip_loss_tot + hparams.disc_wt * perceptual_loss + hparams.mem_wt * (
                        add_loss.mean() + mem_recon.mean())
            loss.backward()
            optimizer.step()

            ### Training disc
            disc_optimizer.zero_grad()

            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
            disc_real_loss.backward()
            pred_te = disc(g_te.detach())
            disc_fake_loss_te = F.binary_cross_entropy(pred_te, torch.zeros((len(pred_te), 1)).to(device))
            pred_tr = disc(g_tr.detach())
            disc_fake_loss_tr = F.binary_cross_entropy(pred_tr, torch.zeros((len(pred_tr), 1)).to(device))
            disc_fake_loss = disc_fake_loss_te + disc_fake_loss_tr
            disc_fake_loss.backward()
            disc_optimizer.step()

            # Logs
            global_step += 1
            running_l1_loss += l1loss.item()
            running_sync_loss += sync_loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()
            running_reconst_loss += mem_recon.mean()
            running_add_loss += add_loss.mean()
            running_lip_loss += lip_loss_tot.item()

            if (step + 1) % (len(train_data_loader) // 8) == 0:
                with torch.no_grad():
                    average_l1_loss, average_sync_loss = eval_model(test_data_loader, global_step, device, [model, dec],
                                                                    disc, syncnet, writer, prefix='valid')
                if average_l1_loss < best_l1_loss or average_sync_loss < best_sync_loss:
                    torch.save({'model_state_dict': model.module.state_dict(),
                                'dec_state_dict': dec.module.state_dict(),
                                'disc_state_dict': disc.module.state_dict()},
                               os.path.join(checkpoint_dir, 'Epoch_%03d_Step_%04d_l1(%.5f)_syn(%.5f).ckpt' % (
                                   global_epoch, global_step, average_l1_loss, average_sync_loss)))

            prog_bar.set_description(
                '{}/{} steps:: L1: {:.4f} Sync: {:.4f} Percep: {:.4f} Fake: {:.4f} recon: {:.4f} add: {:.4f}'.format(
                    step + 1, len(train_data_loader), running_l1_loss / (step + 1),
                    running_sync_loss / (step + 1),
                    running_perceptual_loss / (step + 1),
                    running_disc_fake_loss / (step + 1),
                    running_reconst_loss / (step + 1),
                    running_add_loss / (step + 1)
                ))

            writer.add_scalar('model_/L1_loss_v', l1loss.cpu(), global_step)
            writer.add_scalar('model_/Liploss', lip_loss_tot.cpu(), global_step)
            writer.add_scalar('model_/Sync_loss_v', sync_loss.cpu(), global_step)
            writer.add_scalar('model_/Percep_loss_v', perceptual_loss.cpu(), global_step)
            writer.add_scalar('dis/Fake_loss', disc_fake_loss.cpu(), global_step)
            writer.add_scalar('dis/Real_loss', disc_real_loss.cpu(), global_step)
            writer.add_scalar('dis/Dis_loss', (disc_real_loss + disc_fake_loss).cpu(), global_step)
            writer.add_scalar('model/Total_loss', loss.cpu(), global_step)
            writer.add_scalar('memory/Mem_Add_loss', add_loss.mean().cpu(), global_step)
            writer.add_scalar('memory/Mem_Recon_loss', mem_recon.mean().cpu(), global_step)

            if (step + 1) % (len(train_data_loader) // 8) == 0:
                gen = ((g_te.detach().cpu().numpy().transpose(0, 2, 1, 3, 4) + 1) * 255. / 2).astype(
                    np.uint8)
                gen_collage = np.concatenate([gen[:, i] for i in range(5)], axis=-1)

                gen_o = ((g_tr.detach().cpu().numpy().transpose(0, 2, 1, 3, 4) + 1) * 255. / 2).astype(
                    np.uint8)
                gen_collage_o = np.concatenate([gen_o[:, i] for i in range(5)], axis=-1)

                gt = ((gt.detach().cpu().numpy().transpose(0, 2, 1, 3, 4) + 1) * 255. / 2).astype(
                    np.uint8)
                gt_collage = np.concatenate([gt[:, i] for i in range(5)], axis=-1)  # B, C, H, W*T

                input = ((x.detach().cpu().numpy().transpose(0, 2, 1, 3, 4) + 1) * 255. / 2).astype(
                    np.uint8)
                input_collage = np.concatenate([input[:, i] for i in range(5)], axis=-1)  # B, C, H, W*T
                input_collage1 = input_collage[:, :3, ]
                input_collage2 = input_collage[:, 3:, ]

                writer.add_image('train_0/lip', np.concatenate(
                    [input_collage1[0][::-1, ], input_collage2[0][::-1, ], gen_collage[0][::-1, ],
                     gen_collage_o[0][::-1, ], gt_collage[0][::-1, ]], axis=-2), global_step)
                writer.add_image('train_3/lip', np.concatenate(
                    [input_collage1[3][::-1, ], input_collage2[3][::-1, ], gen_collage[3][::-1, ],
                     gen_collage_o[3][::-1, ], gt_collage[3][::-1, ]], axis=-2), global_step)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
                ax.plot(key_add.cpu().detach().numpy()[3, 3, :])
                writer.add_figure('(3)train_add_3/key_address', fig, global_step)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
                ax.plot(value_add.cpu().detach().numpy()[3, 3, :])
                writer.add_figure('(3)train_add_3/value_address', fig, global_step)


                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
                ax.plot(key_add.cpu().detach().numpy()[0, 2, :])
                writer.add_figure('(0)train_add_2/key_address', fig, global_step)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
                ax.plot(value_add.cpu().detach().numpy()[0, 2, :])
                writer.add_figure('(0)train_add_2/value_address', fig, global_step)

        global_epoch += 1



def eval_model(test_data_loader, global_step, device, model, disc, syncnet, writer, prefix='valid'):
    model, dec = model
    print('Evaluating for {} steps'.format(len(test_data_loader)))
    running_sync_loss, running_l1_loss, running_perceptual_loss, running_reconst_loss, running_add_loss = [], [], [], [], []

    with torch.no_grad():
        for step, (x, indiv_mels, mel, gt, lip) in tqdm(enumerate(test_data_loader)):
            model.eval()
            disc.eval()
            dec.eval()
            syncnet.eval()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            lip = lip.to(device)

            te_fusion, _, feats, mem_recon, add_loss, key_add, value_add, _ = model(indiv_mels, x, lip, False)
            g = dec(te_fusion, feats)

            sync_loss = get_sync_loss(mel, (g + 1) / 2, syncnet)
            perceptual_loss = disc.forward(g)
            perceptual_loss = F.binary_cross_entropy(perceptual_loss, torch.ones((len(perceptual_loss), 1)).cuda())

            l1loss = recon_loss((g + 1) / 2, (gt + 1) / 2)
            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())
            running_perceptual_loss.append(perceptual_loss)
            running_reconst_loss.append(mem_recon.mean())
            running_add_loss.append(add_loss.mean())

    writer.add_scalar(f'model_{prefix}/Eval_Sync_loss', sum(running_sync_loss) / len(running_sync_loss), global_step)
    writer.add_scalar(f'model_{prefix}/Eval_L1_loss', sum(running_l1_loss) / len(running_l1_loss), global_step)
    writer.add_scalar(f'model_{prefix}/Eval_Percep_loss', sum(running_perceptual_loss) / len(running_perceptual_loss),
                      global_step)
    writer.add_scalar(f'memory_{prefix}/Eval_Mem_Add_loss', sum(running_add_loss) / len(running_add_loss),
                      global_step)
    writer.add_scalar(f'memory_{prefix}/Eval_Mem_Recon_loss', sum(running_reconst_loss) / len(running_reconst_loss),
                      global_step)

    print(
        '{}:: L1: {:.4f} Sync: {:.4f} Percep: {:.4f} Mem_recon: {:.4f} Mem_ad: {:.4f}'.format(
            prefix,
            sum(running_l1_loss) / (len(running_l1_loss)),
            sum(running_sync_loss) / (len(running_sync_loss)),
            sum(running_perceptual_loss) / (len(running_perceptual_loss)),
            sum(running_reconst_loss) / len(running_reconst_loss),
            sum(running_add_loss) / len(running_add_loss),
        ))

    gen = ((g.detach().cpu().numpy().transpose(0, 2, 1, 3, 4) + 1) * 255. / 2).astype(np.uint8)  # B, T, C, H, W
    gen_collage = np.concatenate([gen[:, i] for i in range(5)], axis=-1)

    gt = ((gt.detach().cpu().numpy().transpose(0, 2, 1, 3, 4) + 1) * 255. / 2).astype(np.uint8)  # B, T, C, H, W
    gt_collage = np.concatenate([gt[:, i] for i in range(5)], axis=-1)  # B, C, H, W*T

    input = ((x.detach().cpu().numpy().transpose(0, 2, 1, 3, 4) + 1) * 255. / 2).astype(np.uint8)  # B, T, 6, H, W
    input_collage = np.concatenate([input[:, i] for i in range(5)], axis=-1)  # B, C, H, W*T
    input_collage1 = input_collage[:, :3, ]
    input_collage2 = input_collage[:, 3:, ]

    writer.add_image(f'{prefix}_0/lip', np.concatenate(
        [input_collage1[0][::-1, ], input_collage2[0][::-1, ],
         gen_collage[0][::-1, ],
         gt_collage[0][::-1, ]], axis=-2), global_step)
    writer.add_image(f'{prefix}_2/lip', np.concatenate(
        [input_collage1[2][::-1, ], input_collage2[2][::-1, ],
         gen_collage[2][::-1, ],
         gt_collage[2][::-1, ]], axis=-2), global_step)
    writer.add_image(f'{prefix}_3/lip', np.concatenate(
        [input_collage1[3][::-1, ], input_collage2[3][::-1, ],
         gen_collage[3][::-1, ],
         gt_collage[3][::-1, ]], axis=-2), global_step)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
    ax.plot(key_add.cpu().detach().numpy()[0, 2, :])
    writer.add_figure('(0)eval_add_2/key_address', fig, global_step)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
    ax.plot(value_add.cpu().detach().numpy()[0, 2, :])
    writer.add_figure('(0)eval_add_2/value_address', fig, global_step)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
    ax.plot(key_add.cpu().detach().numpy()[2, 2, :])
    writer.add_figure('(2)eval_add_2/key_address', fig, global_step)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
    ax.plot(value_add.cpu().detach().numpy()[2, 2, :])
    writer.add_figure('(2)eval_add_2/value_address', fig, global_step)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
    ax.plot(key_add.cpu().detach().numpy()[3, 2, :])
    writer.add_figure('(3)eval_add_2/key_address', fig, global_step)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[0., hparams.m_slot], ylim=[hparams.min, hparams.max])
    ax.plot(value_add.cpu().detach().numpy()[3, 2, :])
    writer.add_figure('(3)eval_add_2/value_address', fig, global_step)

    return sum(running_l1_loss) / (len(running_l1_loss)), sum(running_sync_loss) / len(running_sync_loss)


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train_dataset = Dataset('train')
    test_dataset = Dataset('test')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=False)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=40,
        num_workers=4, drop_last=True, pin_memory=False)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Wav2Lip()
    dec = Wav2Lip_decoder()
    disc = Wav2Lip_disc_qual()
    syncnet = SyncNet()

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad] + [p for p in dec.parameters() if p.requires_grad],
        lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                                lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])

    load_checkpoint(args.syncnet_checkpoint_path, syncnet)

    model = nn.DataParallel(model).cuda()
    dec = nn.DataParallel(dec).cuda()
    disc = nn.DataParallel(disc).cuda()
    syncnet = nn.DataParallel(syncnet).cuda()
    for p in syncnet.parameters():
        p.requires_grad = False

    train(device, [model, dec], syncnet, disc, train_data_loader, test_data_loader, optimizer,
          disc_optimizer,
          checkpoint_dir=checkpoint_dir,
          nepochs=hparams.nepochs)
