import torch
from torch import nn
from torch.nn import functional as F
from models.memory import Memory
import copy

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class twoD(nn.Module):
    def forward(self, input):
        return input.squeeze(2)


class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.mem = Memory()

        self.lip_encoder = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=2, padding=(0, 3, 3)),
                                                twoD(),
                                                Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                                                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                                                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                                                Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                                                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                                                Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                                                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                                                Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                                Conv2d(512, 512, kernel_size=3, stride=1, padding=1,residual=True),
                                                Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
                                                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
                                                )

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def lip_forward(self, g_te_lip, g_tr_lip):
        B = g_te_lip.size(0)
        input_dim_size = len(g_te_lip.size())
        if input_dim_size > 4:
            g_te_lip = torch.cat([g_te_lip[:, i] for i in range(g_te_lip.size(1))], dim=0)
            g_tr_lip = torch.cat([g_tr_lip[:, i] for i in range(g_tr_lip.size(1))], dim=0)

        g_te_lip_embedding = self.lip_encoder(g_te_lip)
        g_tr_lip_embedding = self.lip_encoder(g_tr_lip)

        return g_te_lip_embedding, g_tr_lip_embedding

    def lip_check(self, g_te_lip, g_tr_lip):
        critic = copy.deepcopy(self.lip_encoder)
        B = g_te_lip.size(0)
        input_dim_size = len(g_te_lip.size())
        if input_dim_size > 4:
            g_te_lip = torch.cat([g_te_lip[:, i] for i in range(g_te_lip.size(1))], dim=0)
            g_tr_lip = torch.cat([g_tr_lip[:, i] for i in range(g_tr_lip.size(1))], dim=0)

        g_te_lip_embedding = critic(g_te_lip)
        g_tr_lip_embedding = critic(g_tr_lip)

        return g_te_lip_embedding, g_tr_lip_embedding


    def forward(self, audio_sequences, face_sequences, lip_sequences, inference):
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
            if not inference:
                lip_sequences = torch.cat([lip_sequences[:, i] for i in range(lip_sequences.size(1))], dim=0)
            else:
                lip_sequences = None

        audio_embedding = self.audio_encoder(audio_sequences)

        if not inference:
            lip_embedding = self.lip_encoder(lip_sequences)
        else:
            lip_embedding = None

        te_fusion, tr_fusion, recon_loss, add_loss, key_add, value_add = self.mem(audio_embedding, lip_embedding, inference)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        te_fusion = torch.split(te_fusion, B, dim=0)
        te_fusion = torch.stack(te_fusion, dim=1)
        if not inference:
            tr_fusion = torch.split(tr_fusion, B, dim=0)
            tr_fusion = torch.stack(tr_fusion, dim=1)

        if not inference:
            lip = torch.split(lip_embedding, B, dim=0)
            lip = torch.stack(lip, dim=1)
            lip = lip[:, 2, :, :, :].squeeze(1)
        else:
            lip = None

        return te_fusion, tr_fusion, feats, recon_loss, add_loss, key_add, value_add, lip


class Wav2Lip_decoder(nn.Module):
    def __init__(self):
        super(Wav2Lip_decoder, self).__init__()

        self.face_decoder_blocks = nn.ModuleList([
        nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

        nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=4, stride=1, padding=0),  # 3,3
                      Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

        nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                      Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                      Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6

        nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                      Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                      Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12

        nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                      Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                      Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24

        nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                      Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                      Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48

        nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                      Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                      Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                        nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                        nn.Tanh())

    def forward(self, audio_embedding, feats):
        B = audio_embedding.size(0)
        audio_embedding = torch.cat([audio_embedding[:, i] for i in range(audio_embedding.size(1))], dim=0)
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            feats.pop()

        x = self.output_block(x)
        x = torch.split(x, B, dim=0)
        outputs = torch.stack(x, dim=2)
        return outputs


class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),
            
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1), 
                                        torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)