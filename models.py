import torch.nn as nn
import torch.nn.functional as F
import torch


class KnowledgeEnrichedEmbedding(nn.Module):
    # takes n entities' embeddings and their descriptions' embeddings
    # output enriched embedding

    # w_out_channels is number of channels of cnn output, not specified in paper.
    # it seems its value should be set to wsize, but actually whatever can.
    # doesn't matter.
    def __init__(self, entity_size=128, word_size=128, word_out_channels=None, enriched_vector_size=128, word_num=20, conv_kernel_size=3, dropout_p=0.4):
        super(KnowledgeEnrichedEmbedding, self).__init__()

        self.e_size = entity_size
        self.w_size = word_size
        self.w_out_channels = self.w_size if word_out_channels is None else word_out_channels
        self.ve_size = enriched_vector_size
        self.word_num = word_num
        self.kernel_size = conv_kernel_size
        self.conv = nn.Conv2d(1, self.w_out_channels, kernel_size=(self.kernel_size, self.w_size), stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.w_out_channels, eps=1e-6)
        self.maxpooling = nn.MaxPool2d((self.word_num - self.kernel_size + 1, 1))
        self.fc = nn.Linear(self.e_size + self.w_size, self.ve_size, bias=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, *inputs):
        entities, words = inputs
        # entity: [batch, n, esize]
        # words: [batch, n, L, wsize]

        batch_size = entities.shape[0]

        # entities = entities.reshape((-1, 1, self.e_size))  # entities: [batch, n, esize] -> [batch * n, 1, esize]
        words = words.reshape((-1, 1, self.word_num, self.w_size))  # word: [batch, n, L, wsize] -> [batch * n, 1, L, wsize]

        conved = F.relu(self.bn(self.conv(words)))  # conved: [batch * n, out_channels, L - ksize + 1, 1]
        pooled = self.maxpooling(conved)  # pooled: [batch * n, out_channels, 1, 1]
        flattened = pooled.reshape((batch_size, -1, self.w_out_channels))  # flattened: [batch, n, out_channels]
        concat = torch.cat([entities, flattened], 2)  # concat: [batch, n, esize + out_channels]

        out = self.dropout(self.fc(concat))  # out: [batch, n, keesize]
        return out


class KernelInteractionModel(nn.Module):
    def __init__(self, K=11, epsilon=1e-6):
        super(KernelInteractionModel, self).__init__()
        self.cos_e = nn.CosineSimilarity(dim=2, eps=epsilon)
        self.cos_w = nn.CosineSimilarity(dim=2, eps=epsilon)
        self.K = K
        self.kernels = []
        self.kernels.append({
            'mu': 1,
            'sigma_sqr': 0.001 ** 2
        })
        for i in range(-9, 10, 2):
            self.kernels.append({
                'mu': i / 10,
                'sigma_sqr': 0.1 ** 2
            })

    def forward(self, *inputs):
        ve, E, W = inputs
        # ve: [batch, 1, keesize]
        # E: [1, ne, keesize]
        # W: [1, nw, wsize]
        cossim_e = self.cos_e(ve, E)  # [batch, ne], cossim_e[i, j] := cos between ve in ith batch and jth entity in doc
        cossim_w = self.cos_w(ve, W)  # [batch, nw]

        phi_e = []
        phi_w = []
        for kernel in self.kernels:
            norms_e = torch.exp(-((cossim_e - kernel['mu']) ** 2) / (2 * kernel['sigma_sqr']))
            phi_e.append(torch.sum(norms_e, dim=1, keepdim=True))  # [batch, 1]

            norms_w = torch.exp(-((cossim_w - kernel['mu']) ** 2) / (2 * kernel['sigma_sqr']))
            phi_w.append(torch.sum(norms_w, dim=1, keepdim=True))  # [batch, 1]

        Phi_e = torch.cat(phi_e, dim=1)  # [batch, K]
        Phi_w = torch.cat(phi_w, dim=1)  # [batch, K]
        Phi = torch.cat([Phi_e, Phi_w], dim=1)  # [batch, 2K]
        return Phi


class KESMSalienceEstimation(nn.Module):
    def __init__(self, entity_size=128, word_size=128, word_out_channels=None, enriched_vector_size=128, word_num=20, conv_kernel_size=3, K=11, epsilon=1e-6, dropout_p=0.4):
        super(KESMSalienceEstimation, self).__init__()
        self.KEE = KnowledgeEnrichedEmbedding(entity_size, word_size, word_out_channels, enriched_vector_size, word_num, conv_kernel_size, dropout_p)
        self.KIM = KernelInteractionModel(K, epsilon)
        self.fc = nn.Linear(2*K, 1, bias=True)
        self.dropout = nn.Dropout(dropout_p)

    def score(self, *inputs):
        doc_E, doc_description, doc_W = inputs
        # doc_E: [ne, esize]
        # doc_description: [ne, L, wsize]
        # doc_W: [nw, wsize]

        doc_E = doc_E.unsqueeze(0)  # doc_E: [ne, esize] -> [1, ne, esize]
        doc_description = doc_description.unsqueeze(0)  # doc_description: [ne, L, wsize] -> [1, ne, L, wsize]
        doc_KEE = self.KEE(doc_E, doc_description)  # doc_KEE: [1, ne, keesize]
        doc_W = doc_W.unsqueeze(0)  # doc_W: [nw, wsize] -> [1, nw, wsize]

        query_KEE = doc_KEE.squeeze(0).unsqueeze(1)  # query_E: [ne, 1, keesize]

        query_Phi = self.KIM(query_KEE, doc_KEE, doc_W)  # query_Phi: [ne, 2K]
        query_score = self.fc(query_Phi)  # query_score: [ne, 1]
        return query_score

    def forward(self, *inputs):
        pos_E, pos_description, neg_E, neg_description, doc_E, doc_description, doc_W = inputs
        # pos_E: [batch, 1, esize]
        # pos_description: [batch, 1, L, wsize]
        # neg_E: [batch, 1, esize]
        # neg_description: [batch, 1, L, wsize]
        # doc_E: [ne, esize]
        # doc_description: [ne, L, wsize]
        # doc_W: [nw, wsize]


        # doc_E: [ne, esize]
        # doc_description: [ne, L, wsize]
        # doc_W: [nw, wsize]
        # pos: [npos, esize]
        # pos_description: [npos, L, wsize]
        # neg: [nneg, esize]
        # neg_description: [nneg, L, wsize]


        # calculate doc's entities' embeddings
        doc_E = doc_E.unsqueeze(0)  # doc_E: [ne, esize] -> [1, ne, esize]
        doc_description = doc_description.unsqueeze(0)  # doc_description: [ne, L, wsize] -> [1, ne, L, wsize]
        doc_KEE = self.KEE(doc_E, doc_description)  # doc_KEE: [1, ne, keesize]

        # calculate query entities' embeddings
        pos_KEE = self.KEE(pos_E, pos_description)  # pos_KEE: [batch, 1, keesize]
        neg_KEE = self.KEE(neg_E, neg_description)  # neg_KEE: [batch, 1, keesize]

        # KIM
        doc_W = doc_W.unsqueeze(0)  # doc_W: [nw, wsize] -> [1, nw, wsize]
        pos_Phi = self.KIM(pos_KEE, doc_KEE, doc_W)  # pos_Phi: [batch, 2K]
        neg_Phi = self.KIM(neg_KEE, doc_KEE, doc_W)  # neg_Phi: [batch, 2K]

        out_pos = self.dropout(self.fc(pos_Phi))  # pos_out: [batch, 1]
        out_neg = self.dropout(self.fc(neg_Phi))  # neg_out: [batch, 1]

        out_pos = torch.tanh(out_pos)  # pos_out: [batch, 1]
        out_neg = torch.tanh(out_neg)  # neg_out: [batch, 1]

        return out_pos, out_neg

        #
        # KEE_embedding_list = []
        # ne = E.shape[1]
        # for i in range(ne):
        #     KEE_embedding_list.append(self.KEE(E[:, i, :], description_E[:, i, :, :]))
        #
        # KEE = torch.cat(KEE_embedding_list, dim=1)  # KEE: [batch, ne, keesize]
        # ve_pos = torch.index_select(KEE, 1, index_pos)  # ve_pos: [batch, npos, keesize]
        # ve_neg = torch.index_select(KEE, 1, index_neg)  # ve_neg: [batch, nneg, keesize]




        #
        #
        # KEE = self.KEE(E, description_E)  # KEE: [batch, ne, keesize]
        # query_ve = KEE[:, query_index, :]  # query_ve: [batch, 1, keesize]
        # KIM = self.KIM(query_ve, E, W)  # KIM: [batch, 2K]
        #


        #
        #
        # x_pos, x_neg = inputs
        # out_pos = self.fc(x_pos)
        # out_neg = self.fc(x_neg)
        # return out_pos, out_neg
