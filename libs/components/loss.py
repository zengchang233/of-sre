import math
from numpy.lib.arraysetops import isin

#  import torch
#  import torch.nn as nn
#  import torch.nn.functional as F
import oneflow as of
import oneflow.nn as nn
import oneflow.nn.functional as F

#  from libs.components.scores import CosineScore

#  class GE2ELoss(nn.Module):
#
#      def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
#          '''
#          Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
#          Accepts an input of size (N, M, D)
#              where N is the number of speakers in the batch,
#              M is the number of utterances per speaker,
#              and D is the dimensionality of the embedding vector (e.g. d-vector)
#          Args:
#              - init_w (float): defines the initial value of w in Equation (5) of [1]
#              - init_b (float): definies the initial value of b in Equation (5) of [1]
#          '''
#          super(GE2ELoss, self).__init__()
#          self.w = nn.Parameter(torch.tensor(init_w))
#          self.b = nn.Parameter(torch.tensor(init_b))
#          self.loss_method = loss_method
#
#          assert self.loss_method in ['softmax', 'contrast']
#
#          if self.loss_method == 'softmax':
#              self.embed_loss = self.embed_loss_softmax
#          if self.loss_method == 'contrast':
#              self.embed_loss = self.embed_loss_contrast
#
#      def calc_new_centroids(self, dvecs, centroids, spkr, utt):
#          '''
#          Calculates the new centroids excluding the reference utterance
#          '''
#          excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
#          excl = torch.mean(excl, 0)
#          new_centroids = []
#          for i, centroid in enumerate(centroids):
#              if i == spkr:
#                  new_centroids.append(excl)
#              else:
#                  new_centroids.append(centroid)
#          return torch.stack(new_centroids)
#
#      def calc_cosine_sim(self, dvecs, centroids):
#          '''
#          Make the cosine similarity matrix with dims (N,M,N)
#          '''
#          cos_sim_matrix = []
#          for spkr_idx, speaker in enumerate(dvecs):
#              cs_row = []
#              for utt_idx, utterance in enumerate(speaker):
#                  new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
#                  # vector based cosine similarity for speed
#                  cs_row.append(torch.clamp(torch.mm(utterance.unsqueeze(1).transpose(0,1), new_centroids.transpose(0,1)) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
#              cs_row = torch.cat(cs_row, dim=0)
#              cos_sim_matrix.append(cs_row)
#          return torch.stack(cos_sim_matrix)
#
#      def embed_loss_softmax(self, dvecs, cos_sim_matrix):
#          '''
#          Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
#          '''
#          N, M, _ = dvecs.shape
#          L = []
#          for j in range(N):
#              L_row = []
#              for i in range(M):
#                  L_row.append(-F.log_softmax(cos_sim_matrix[j,i], 0)[j])
#              L_row = torch.stack(L_row)
#              L.append(L_row)
#          return torch.stack(L)
#
#      def embed_loss_contrast(self, dvecs, cos_sim_matrix):
#          '''
#          Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
#          '''
#          N, M, _ = dvecs.shape
#          L = []
#          for j in range(N):
#              L_row = []
#              for i in range(M):
#                  centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j,i])
#                  excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j+1:]))
#                  L_row.append(1. - torch.sigmoid(cos_sim_matrix[j,i,j]) + torch.max(excl_centroids_sigmoids))
#              L_row = torch.stack(L_row)
#              L.append(L_row)
#          return torch.stack(L)
#
#      def forward(self, dvecs):
#          '''
#          Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
#          '''
#          #Calculate centroids
#          centroids = torch.mean(dvecs, 1)
#
#          #Calculate the cosine similarity matrix
#          cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
#          print(cos_sim_matrix.shape)
#          torch.clamp(self.w, 1e-6)
#          cos_sim_matrix = cos_sim_matrix * self.w + self.b
#          L = self.embed_loss(dvecs, cos_sim_matrix)
#          return L.sum()
#
#  class GE2E(nn.Module):
#      def __init__(self):
#          super(GE2E, self).__init__()
#          self.w = nn.Parameter(torch.Tensor(1, 1), requires_grad = True)
#          self.b = nn.Parameter(torch.Tensor(1, 1), requires_grad = True)
#          self.cosine_score = CosineScore()
#          self._init()
#
#      def _init(self):
#          nn.init.kaiming_normal_(self.w)
#          nn.init.kaiming_normal_(self.b)
#
#      def center(self, embedding):
#          num_spks, num_utts, dim = embedding.size()
#          x = embedding.repeat(1, num_utts, 1) # 重复n_utts次
#          mask = torch.logical_not(torch.eye(num_utts)).repeat(num_spks, 1).view(-1).to(x.device) # mask掉本身
#          masked_x = x.view(-1, dim)[mask].contiguous().view(num_spks * num_utts, -1, dim) # 每n_spks个划分为一组，分别对应mask掉的部分
#          center = masked_x.mean(dim = 1)
#          return center
#
#      def get_prob(self, score):
#          return self.w * score + self.b, score
#
#      def forward(self, embedding):
#          '''
#          embedding: N, M, D (num_spks, num_utts, dim)
#          '''
#          center = self.center(embedding) # N * M, D
#          num_spks, num_utts, dim = embedding.size()
#          score_matrix = self.cosine_score(embedding.view(-1, dim), center)
#          mask = torch.eye(num_utts, dtype = torch.bool).repeat(num_spks, num_spks).to(embedding.device) # mask
#          score_matrix = score_matrix.view(-1)[mask.view(-1)].view(num_spks * num_utts, -1)
#          score_matrix, _ = self.get_prob(score_matrix)
#          loss_mask_matrix = torch.eye(mask.size(0), dtype = torch.long).to(embedding.device) # 候选单位阵
#          loss_mask = loss_mask_matrix.view(-1)[mask.view(-1)] # .view(-1, 1)
#          loss = -F.log_softmax(score_matrix, dim = 1).view(-1)
#          loss = loss[loss_mask.bool()]
#          return loss.sum()
#
#  class GE2EBackend(nn.Module):
#      def __init__(self):
#          super(GE2EBackend, self).__init__()
#          self.w = nn.Parameter(torch.Tensor(1, 1))
#          self.b = nn.Parameter(torch.Tensor(1, 1))
#          self.prob = nn.Sigmoid()
#
#          self.bce = nn.BCELoss(reduction = 'sum')
#          self._init()
#
#      def _init(self):
#          nn.init.kaiming_normal_(self.w)
#          nn.init.kaiming_normal_(self.b)
#
#      def get_prob(self, score):
#          score_matrix = self.w * score + self.b # scaled cosine score
#          score_matrix = self.prob(score_matrix) # sigmoid prob
#          return score_matrix, score
#
#      def forward(self, embedding, score_matrix, ground_truth, hard_num = None):
#          '''
#          score_matrix: num_spks * num_utts * num_spks
#          '''
#          num_spks, num_utts, _ = embedding.size()
#          score_matrix, _ = self.get_prob(score_matrix)
#          loss = -F.log_softmax(score_matrix.view(num_spks * num_utts, -1), dim = 1).view(-1)
#          mask = ground_truth.bool()
#          loss = loss[mask]
#          if isinstance(hard_num, int):
#              loss, _ = torch.topk(loss, k = hard_num)
#          loss = loss.sum()
#          labels = ground_truth.view(-1, 1).float()
#          bce_loss = self.bce(score_matrix, labels)
#          # loss = bce_loss + 0.1 * loss
#          return loss, bce_loss, score_matrix.view(-1, 1)
#
#  class NoiseConEstLoss(nn.Module):
#      def __init__(self):
#          super(NoiseConEstLoss, self).__init__()
#
#      def forward(self, scores, labels):
#          pass
#
#  class SupConLoss(nn.Module):
#      """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#      It also supports the unsupervised contrastive loss in SimCLR"""
#      def __init__(self, temperature=0.07, contrast_mode='all',
#                   base_temperature=0.07):
#          super(SupConLoss, self).__init__()
#          self.temperature = temperature
#          self.contrast_mode = contrast_mode
#          self.base_temperature = base_temperature
#
#      def get_prob(self, features, labels):
#          pass
#
#      def forward(self, features, labels=None, mask=None):
#          """Compute loss for model. If both `labels` and `mask` are None,
#          it degenerates to SimCLR unsupervised loss:
#          https://arxiv.org/pdf/2002.05709.pdf
#          Args:
#              features: hidden vector of shape [bsz, n_views, ...].
#              labels: ground truth of shape [bsz].
#              mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                  has the same class as sample i. Can be asymmetric.
#          Returns:
#              A loss scalar.
#          """
#          device = (torch.device('cuda')
#                    if features.is_cuda
#                    else torch.device('cpu'))
#
#          if len(features.shape) < 3:
#              raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                               'at least 3 dimensions are required')
#          if len(features.shape) > 3:
#              features = features.view(features.shape[0], features.shape[1], -1)
#
#          batch_size = features.shape[0]
#          if labels is not None and mask is not None:
#              raise ValueError('Cannot define both `labels` and `mask`')
#          elif labels is None and mask is None:
#              mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#          elif labels is not None:
#              labels = labels.contiguous().view(-1, 1)
#              if labels.shape[0] != batch_size:
#                  raise ValueError('Num of labels does not match num of features')
#              mask = torch.eq(labels, labels.T).float().to(device)
#          else:
#              mask = mask.float().to(device)
#
#          contrast_count = features.shape[1]
#          contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#          if self.contrast_mode == 'one':
#              anchor_feature = features[:, 0]
#              anchor_count = 1
#          elif self.contrast_mode == 'all':
#              anchor_feature = contrast_feature
#              anchor_count = contrast_count
#          else:
#              raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
#
#          # compute logits
#          anchor_dot_contrast = torch.div(
#              torch.matmul(anchor_feature, contrast_feature.T),
#              self.temperature)
#          # for numerical stability
#          logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#          logits = anchor_dot_contrast - logits_max.detach()
#
#          # tile mask
#          mask = mask.repeat(anchor_count, contrast_count)
#          # mask-out self-contrast cases
#          logits_mask = torch.scatter(
#              torch.ones_like(mask),
#              1,
#              torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#              0
#          )
#          mask = mask * logits_mask
#
#          # compute log_prob
#          exp_logits = torch.exp(logits) * logits_mask
#          log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#
#          # compute mean of log-likelihood over positive
#          mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#
#          # loss
#          loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#          loss = loss.view(anchor_count, batch_size).mean()
#
#          return loss
#
#  class BCELoss(nn.Module):
#      def __init__(self, input_dim, num_classes = 1, affine = True, reduction = 'mean'):
#          super(BCELoss, self).__init__()
#          self.input_dim = input_dim
#          self.num_classes = num_classes
#          self.affine = affine
#          if self.affine:
#              self.logistic = nn.Linear(input_dim, num_classes)
#          self.sigmoid = nn.Sigmoid()
#          self.bce_loss = nn.BCELoss(reduction = reduction)
#
#      def get_prob(self, inputs):
#          if self.affine:
#              logits = self.logistic(inputs)
#              prob = self.sigmoid(logits)
#              #  return prob, logits
#              return logits, prob
#          else:
#              prob = self.sigmoid(inputs)
#              return inputs, prob
#
#      def forward(self, inputs, labels):
#          labels = labels.view(-1, 1).float()
#          #  prob, _ = self.get_prob(inputs)
#          _, prob = self.get_prob(inputs)
#          loss = self.bce_loss(prob, labels)
        #  return loss, prob

class CrossEntropy(nn.Module):
    def __init__(self, embedding_size, num_classes, reduction = 'mean'):
        super(CrossEntropy, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.reduction = reduction
        self.fc = nn.Linear(embedding_size, num_classes)

    def get_prob(self, inputs):
        logits = self.fc(inputs)
        scores = F.softmax(logits, dim = 1)
        return scores[:, 1], logits

    def forward(self, embeddings, labels):
        logits = self.fc(embeddings)
        loss = F.cross_entropy(logits, labels, reduction = self.reduction)
        return loss, logits

class ASoftmax(nn.Module):
    def __init__(self, embedding_size, num_classes, margin):
        super(ASoftmax, self).__init__()

    def forward(self, embeddings, labels):
        pass

class AMSoftmax(nn.Module):
    def __init__(self, embedding_size, num_classes, s, margin):
        super(AMSoftmax, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.margin = margin
        self.weights = nn.Parameter(of.Tensor(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.weights)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        logits = F.linear(F.l2_normalize(embeddings, dim = 1), F.l2_normalize(self.weights, dim = 1))
        margin = of.zeros_like(logits)
        #  margin.scatter_(1, labels.view(-1,1), self.margin)
        margin = of.scatter(margin, 1, labels.view(-1, 1), self.margin)
        m_logits = self.s * (logits - margin)
        loss = self.cross_entropy(m_logits, labels)
        return loss, logits

class LMCL(AMSoftmax):
    def __init__(self, embedding_size, num_classes, s, margin):
        super(LMCL, self).__init__(embedding_size, num_classes, s, margin)

    def forward(self, embeddings, labels):
        super().forward(embeddings, labels)

class AAMSoftmax(nn.Module):
    def __init__(self):
        super(AAMSoftmax, self).__init__()

    def forward(self, embeddings, labels):
        pass

#  class OnlineTripletLoss(nn.Module):
    #  def __init__(self, centers, margin, selector = 'hardest'):
    #      super(OnlineTripletLoss, self).__init__()
    #      self.margin = margin
    #      self.centers = F.normalize(centers)
    #      self.centers.requires_grad = False
    #      self.selector = selector
    #
    #  def forward(self, embeddings, labels):
    #      embeddings = embeddings.cpu()
    #      cos_matrix = F.linear(embeddings, self.centers)# cos_matrix batch_size * 1211
    #      rows = torch.arange(embeddings.size(0))
    #      positive_cos = cos_matrix[rows, labels].view(-1,1) # 32 * 1
    #      idx = torch.ones((embeddings.size(0), self.centers.size(0)), dtype = rows.dtype) # 32 * 1211
    #      idx[rows, labels] = 0
    #      negative_cos_matrix = cos_matrix[idx > 0].view(embeddings.size(0), -1) # 32 * 1210
    #      loss_values = negative_cos_matrix + self.margin - positive_cos # 求出所有的loss 32 * 1210
    #      if self.selector == 'hardest': # 挑选出最大的loss
    #          loss_value, _ = torch.max(loss_values, dim = 1)
    #      if self.selector == 'hard':
    #          pass
    #      if self.selector == 'semihard':
    #          pass
    #      losses = F.relu(loss_value.view(-1,1))
#          return losses.mean(), (loss_value > 0).sum().item()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_cos = F.cosine_similarity(embeddings[triplets[:,0]], embeddings[triplets[:,1]])  # .pow(.5)
        an_cos = F.cosine_similarity(embeddings[triplets[:,0]], embeddings[triplets[:,2]]) # .pow(.5)
        losses = F.relu(an_cos - ap_cos + self.margin)
        return losses.mean(), len(triplets)

class OnlineContrastiveLoss(nn.Module):
    def __init__(self, margin, pairs_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pairs_selector = pairs_selector

    def forward(self, embeddings, target):
        pass

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, embeddings, labels):
        pass


if __name__ == '__main__':
    pass
    # ocsoftmax = OCSoftmax(64)
    # inputs = torch.randn(4, 64)
    # target = torch.tensor([0,0,1,0], dtype = torch.int64)
    # output = ocsoftmax(inputs, target)
    # print(output)
    #  ge2e = GE2E()
    #  a = torch.randn(4,5,8, requires_grad=True)
    #  loss = ge2e(a)
    #  loss.backward()
