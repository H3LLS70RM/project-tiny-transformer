import random

class SyntheticICLDataset:
    """
    Generates synthetic ICL sequences for tasks:
    - addition
    - function mapping
    - decoding
    """

    def __init__(self, task="addition", n_samples=200000, n_context=3, addition_range=(0, 99),  motif_range=(5, 10), mapping_range=(1, 10), mapping_b_range=(0, 999), mapping_fn=lambda a,b,x: a*x + b, noise_ratio=0.0):
        self.task = task
        self.n_samples = n_samples
        self.n_context = n_context
        self.addition_range = addition_range
        self.mapping_fn = mapping_fn
        self.motif_range = motif_range
        self.mapping_range = mapping_range
        self.mapping_b_range = mapping_b_range
        self.noise_ratio = noise_ratio

    def generate_example(self, apply_noise=False, a=None, b=None):
        if self.task == "addition":
            # Generate random addition problem
            a = random.randint(*self.addition_range)
            b = random.randint(*self.addition_range)
            ans = a + b
            if apply_noise:
                # generate a wrong answer
                wrong = random.randint(*self.addition_range) + random.randint(*self.addition_range)
                while wrong == ans:
                    wrong = random.randint(*self.addition_range) + random.randint(*self.addition_range)
                ans = wrong
            return f"{a} + {b} = {ans}" 

        elif self.task == "mapping":
            # Generate random linear function
            x = random.randint(*self.mapping_range)
            y = self.mapping_fn(a, b, x)
            if apply_noise:
                wrong_y = random.randint(-100, 1000)
                while wrong_y == y:
                    wrong_y = random.randint(-100, 1000)
                y = wrong_y
            return f"{str(x)} -> {str(y)}"

        elif self.task == "decoding":
            # Substitution Cipher (Single Example)
            chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            k = random.choice(chars)
            v = random.choice(chars)
            if apply_noise:
                wrong_chars = [c for c in chars if c != v]
                v = random.choice(wrong_chars)
            return f"{k} -> {v}"

        else:
            raise ValueError("Invalid task.")

    def generate_sequence(self, return_answer=False):
        # Determine if noise should be applied to the sequence
        apply_noise = random.random() < self.noise_ratio
        # Determine current context size
        if isinstance(self.n_context, tuple) or isinstance(self.n_context, list):
             current_n_context = random.randint(self.n_context[0], self.n_context[1])
        else:
             current_n_context = self.n_context

        # # Generate longer, structured context using generic generator
        # examples = [self.generate_example(apply_noise=(random.random() < self.noise_ratio)) for _ in range(current_n_context)]

        if self.task == "addition":
            examples = []
            a = 0
            b = 0
            while len(examples) < current_n_context:
                # Generate random addition problem
                a = random.randint(*self.addition_range)
                b = random.randint(*self.addition_range)
                ans = a + b
                if apply_noise:
                    # generate a wrong answer
                    wrong = random.randint(*self.addition_range) + random.randint(*self.addition_range)
                    while wrong == ans:
                        wrong = random.randint(*self.addition_range) + random.randint(*self.addition_range)
                    ans = wrong
                if(str(a) in [e.split(" + ")[0] for e in examples] and str(b) in [e.split(" + ")[1] for e in examples]):
                    continue
                examples.append(f"{a} + {b} = {ans}")
            while(True):
                a = random.randint(*self.addition_range)
                b = random.randint(*self.addition_range)
                if(str(a) in [e.split(" + ")[0] for e in examples] and str(b) in [e.split(" + ")[1] for e in examples]):
                    continue
                break
            query = f"{a} + {b} = "
            answer = str(a + b) + "\n"
        elif self.task == "mapping":
            a = random.randint(*self.mapping_range)
            b = random.randint(*self.mapping_b_range)
            
            # Generate context examples adhering to this function
            examples = []
            while len(examples) < current_n_context:
                x = random.randint(*self.mapping_range)
                y = self.mapping_fn(a, b, x)
                if random.random() < self.noise_ratio:
                    wrong_y = random.randint(-100, 1000)
                    while wrong_y == y:
                        wrong_y = random.randint(-100, 1000)
                    y = wrong_y
                if(str(x) in [e.split(" -> ")[0] for e in examples]):
                    continue
                examples.append(f"{x} -> {y}")
            # for _ in range(current_n_context):
            #     x = random.randint(*self.mapping_range)
            #     y = self.mapping_fn(a, b, x)
            #     if random.random() < self.noise_ratio:
            #         wrong_y = random.randint(-100, 1000)
            #         while wrong_y == y:
            #             wrong_y = random.randint(-100, 1000)
            #         y = wrong_y
            #     examples.append(f"{x} -> {y}")
            
            # Generate query that is not in the examples
            while(True):
                x = random.randint(*self.mapping_range)
                y = self.mapping_fn(a, b, x)
                if(str(x) in [e.split(" -> ")[0] for e in examples]):
                    continue
                break
            query = f"{x} -> "
            answer = str(y) + "\n"
        elif self.task == "decoding":
            # Substitution Cipher
            chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            shuffled = chars.copy()
            random.shuffle(shuffled)
            cipher_map = {k: v for k, v in zip(chars, shuffled)}
            
            examples = []
            if current_n_context > 0:
                # Pick exactly one unique key per context shot
                num_keys = min(current_n_context, len(chars))
                seq_keys = random.sample(chars, num_keys)
                
                for k in seq_keys:
                    v = cipher_map[k]
                    if random.random() < self.noise_ratio:
                        wrong_chars = [c for c in chars if c != v]
                        v = random.choice(wrong_chars)
                    examples.append(f"{k} -> {v}")
                
                # Query must be one of the keys shown in the context
                query_k = random.choice(seq_keys)
            else:
                query_k = random.choice(chars)
                
            query = f"{query_k} -> "
            answer = cipher_map[query_k] + "\n"
        else:
            raise ValueError(f"Invalid task: {self.task}")
        prompt = "\n".join(examples) + "\n" + query
        if return_answer:
            return {"prompt": prompt, "answer": answer}
        else:
            return prompt

    def build_dataset(self, return_answer=False):
        data = []
        while len(data) < self.n_samples:
            seq = self.generate_sequence(return_answer=return_answer)
            if seq is not None:
                data.append(seq)
        return data