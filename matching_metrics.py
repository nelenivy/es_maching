from matching_utils import sort_pair, sort_pair_with_dist


class PairsClassificationQuality:
    def process(self, pairs_in, golden_set, all_pairs_num, prefix_length = 0, sort_ids_in_pair=True):
        self.sort_ids_in_pair = sort_ids_in_pair
        self.golden_set = golden_set
        self.prefix_length = prefix_length
        self.final_set = set()
        
        pairs = sorted(pairs_in, key=lambda elem: elem[2])
        self.lgs = len(self.golden_set) 
        self.neg_pairs_num = all_pairs_num - self.lgs
        self.prev_dist = -10000000        
        self.max_f1 = -1
        self.thr = -10000
        self.topn = -1

        self.precs = []
        self.recalls = []
        self.f1s = []
        self.tps = []
        self.fps = []
        self.threshes = []
        self.tp = 0
        self.fp = 0
        
        self.valid_cases = [set(), set()]
        self.prec_top_k = [[], []]
        self.f1_top_k = [[], []]
        
        print(len(pairs))
        
        for i,pair_in in enumerate(pairs):
            pair, curr_dist = self.prepare_pair(pair_in)
            
            for j in range(len(pair)):
                self.valid_cases[j].add(pair[j])
                
            if pair in self.final_set:
                continue
            
            self.process_prev(curr_dist)
            self.process_curr_pair(pair, curr_dist)
        
        self.process_prev(curr_dist)
        
        return (self.pp, self.rr, self.max_f1), self.thr, self.topn
    
    def process_curr_pair(self, pair, curr_dist):
        self.prev_dist = curr_dist
        curr_class = int(pair in self.golden_set)
        self.tp += curr_class
        self.fp += 1 - curr_class
        self.final_set.add(pair)
        
    def process_prev(self, curr_dist):
        if self.final_set and (self.prev_dist != curr_dist):
                p = float(self.tp)/len(self.final_set)
                r = float(self.tp)/self.lgs
                f1 = 2*p*r/(p+r) if (p*r>0) else 0
                self.precs.append(p)
                self.recalls.append(r)
                self.f1s.append(f1)
                self.tps.append(r)
                self.fps.append(float(self.fp) / self.neg_pairs_num)
                self.threshes.append(self.prev_dist)
                
                for j in range(len(self.prec_top_k)):
                    self.prec_top_k[j].append(r * self.lgs / len(self.valid_cases[j]))
                    self.f1_top_k[j].append(2 * (self.prec_top_k[j][-1] * r) / (self.prec_top_k[j][-1] + r + 0.0000000001) )
                    
                if f1 > self.max_f1 and len(self.final_set) > self.prefix_length:
                    self.max_f1 = f1
                    self.pp = p
                    self.rr = r
                    self.thr = self.prev_dist
                    self.topn = len(self.final_set)
                    

    def prepare_pair(self, pair_in):
        if self.sort_ids_in_pair:
            pair_with_dist = sort_pair_with_dist(pair_in)
        else:
            pair_with_dist = pair_in

        pair = pair_with_dist[:2]
        curr_dist = pair_with_dist[2]

        return pair, curr_dist  