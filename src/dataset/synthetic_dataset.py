import random

class SyntheticICLDataset:
    """
    Generates synthetic ICL sequences for tasks:
    - addition (simple baseline)
    - arithmetic (hard ICL variant)
    - arithmetic_shuffled (context-dependent operator meanings)
    - arithmetic_symbolic (prompt-defined symbolic arithmetic)
    - function mapping
    - decoding
    """

    def __init__(self, task="addition", n_samples=200000, n_context=3, addition_range=(0, 99), motif_range=(5, 10), mapping_range=(1, 10), mapping_a_range=(1, 15), mapping_b_range=(0, 99), mapping_fn=lambda a,b,x: a*x + b, noise_ratio=0.0, is_ood=False, digit_level=False, rule_diversity=False, mapping_ood_type='exponential', decoding_reversal=False, hard_icl=False, rule_family_idx=None, exclude_family_idx=None, allow_repeats=True):
        self.task = task
        self.n_samples = n_samples
        self.n_context = n_context
        self.addition_range = addition_range
        self.mapping_fn = mapping_fn
        self.motif_range = motif_range
        self.mapping_range = mapping_range
        self.mapping_a_range = mapping_a_range
        self.mapping_b_range = mapping_b_range
        self.noise_ratio = noise_ratio
        self.is_ood = is_ood
        self.digit_level = digit_level
        self.rule_diversity = rule_diversity
        self.mapping_ood_type = mapping_ood_type
        self.decoding_reversal = decoding_reversal
        self.hard_icl = hard_icl
        self.rule_family_idx = rule_family_idx
        self.exclude_family_idx = exclude_family_idx
        self.allow_repeats = allow_repeats
        
        # Branching logic for the two addition variants
        if self.task == "arithmetic":
            self.hard_icl = True
        elif self.task == "addition":
            self.hard_icl = False

    rule_families = [
        lambda x, a, b: a * x + b,          # 0: Linear (Standard)
        lambda x, a, b: a * (x + b),        # 1: Shifted Linear
        lambda x, a, b: (x // a) + b,       # 2: Division mapping
        lambda x, a, b: abs(a * x - b),     # 3: Difference
        lambda x, a, b: (a * x) % 100,      # 4: Modulo mapping
    ]

    symbolic_rule_families = [
        lambda x, y: x + y,            # 0: Addition
        lambda x, y: x - y,            # 1: Subtraction
        lambda x, y: max(x, y),        # 2: Max
        lambda x, y: min(x, y),        # 3: Min
        lambda x, y: abs(x - y),       # 4: Absolute Difference
        lambda x, y: (x + y) % 10,     # 5: Modulo Addition
        lambda x, y: x * y,            # 6: Multiplication
        lambda x, y: x ^ y,            # 7: XOR
        lambda x, y: x | y,            # 8: OR
        lambda x, y: x & y,            # 9: AND
    ]
    
    def generate_sequence(self, return_answer=False):
        # Determine if noise should be applied to the sequence
        apply_noise = random.random() < self.noise_ratio
        # Determine current context size
        if isinstance(self.n_context, tuple) or isinstance(self.n_context, list):
             current_n_context = random.randint(self.n_context[0], self.n_context[1])
        else:
             current_n_context = self.n_context
             
        # Determine global separator for this sequence (Jitter in arithmetic/Hard ICL)
        if (self.hard_icl and self.task != "addition") or self.task in ("arithmetic", "arithmetic_symbolic", "arithmetic_shuffled"):
            sep = random.choice(["->", ":", "==", "=>", "="])
            sep = f" {sep} " if sep != ":" else ":"
        else:
            sep = "->" if self.task in ("mapping", "decoding") else "="
            sep = f" {sep} "
 
        # Generate examples in accordance with the task
        if self.task == "addition":
            used_pairs = set()
            examples = []
            
            # Base range for addition/subtraction
            base_range = self.addition_range
            
            def format_val(v):
                if self.digit_level:
                    return " ".join(str(v))
                return str(v)
            
            # Hard ICL: Symbolic Addition (Operator selection per sequence)
            op_char = "+"
            op_fn = lambda x, y: x + y

            # Use out-of-distribution range if requested (Relative to operator base_range)
            current_range = base_range
            if self.is_ood:
                # Set range to be strictly above the training range
                shift = (base_range[1] - base_range[0]) + 10
                current_range = (base_range[1] + shift, base_range[1] + shift + 100)
 
            while len(examples) < current_n_context:
                a = random.randint(*current_range)
                b = random.randint(*current_range)
                if (a, b) in used_pairs:
                    continue
                used_pairs.add((a, b))
 
                ans = op_fn(a, b)
                if apply_noise:
                    wrong = random.randint(-100, 1000)
                    while wrong == ans:
                        wrong = random.randint(-100, 1000)
                    ans = wrong
                examples.append(f"{format_val(a)} {op_char} {format_val(b)}{sep}{format_val(ans)}")
            
            while True:
                a = random.randint(*current_range)
                b = random.randint(*current_range)
                if (a, b) in used_pairs:
                    continue
                break
            query = f"{format_val(a)} {op_char} {format_val(b)}{sep}"
            answer = format_val(op_fn(a, b)) + "\n"
        elif self.task == "arithmetic_symbolic":
            used_pairs = set()
            examples = []
            symbol_pool = list("@#$%^&*?!=+~<>:;{}[]")
            op_symbol = random.choice(symbol_pool)

            if self.exclude_family_idx is not None:
                available = [i for i in range(len(self.symbolic_rule_families)) if i != self.exclude_family_idx]
                family_idx = random.choice(available)
            else:
                family_idx = random.randint(0, len(self.symbolic_rule_families) - 1)
            
            rule_family = self.symbolic_rule_families[family_idx]

            current_range = (0, 9)
            if self.is_ood:
                current_range = (10, 19)

            def format_val(v):
                if self.digit_level:
                    return " ".join(str(v))
                return str(v)

            def symbolic_fn(x, y):
                return rule_family(x, y)

            while len(examples) < current_n_context:
                x = random.randint(*current_range)
                y = random.randint(*current_range)
                if (x, y) in used_pairs:
                    continue
                used_pairs.add((x, y))

                ans = symbolic_fn(x, y)
                if apply_noise:
                    wrong = random.randint(-9, 100)
                    while wrong == ans:
                        wrong = random.randint(-9, 100)
                    ans = wrong
                examples.append(f"{format_val(x)} {op_symbol} {format_val(y)}{sep}{format_val(ans)}")

            while True:
                x = random.randint(*current_range)
                y = random.randint(*current_range)
                if not self.allow_repeats and (x, y) in used_pairs:
                    continue
                break

            query = f"{format_val(x)} {op_symbol} {format_val(y)}{sep}"
            answer = format_val(symbolic_fn(x, y)) + "\n"
        elif self.task == "arithmetic_shuffled":
            used_pairs = set()
            examples = []
            
            # Base range for addition/subtraction
            base_range = (0, 100)
            
            def format_val(v):
                if self.digit_level:
                    return " ".join(str(v))
                return str(v)
            
            # Map of standard symbols to functions
            all_ops = {
                "+": lambda x, y: x + y,
                "-": lambda x, y: x - y,
                "*": lambda x, y: x * y,
                "max": lambda x, y: max(x, y),
                "min": lambda x, y: min(x, y),
            }
            
            symbols = ["+", "-", "*", "max", "min"]
            functions = [all_ops[s] for s in symbols]
            
            # Randomly permute the functions for THIS sequence
            shuffled_funcs = functions.copy()
            random.shuffle(shuffled_funcs)
            symbol_to_fn = {s: f for s, f in zip(symbols, shuffled_funcs)}
            
            # Pick ONE symbolic operator to test in this sequence
            target_symbol = random.choice(symbols)
            op_fn = symbol_to_fn[target_symbol]
            
            # Ranges to keep results manageable
            if target_symbol == "*" or op_fn == all_ops["*"]:
                base_range = (0, 20)
            
            current_range = base_range
            if self.is_ood:
                current_range = (base_range[1] + 1, base_range[1] + 50)

            while len(examples) < current_n_context:
                a = random.randint(*current_range)
                b = random.randint(*current_range)
                if (a, b) in used_pairs:
                    continue
                used_pairs.add((a, b))

                ans = op_fn(a, b)
                if apply_noise:
                    # Inject label noise: replace correct answer with a random wrong one
                    wrong = random.randint(-50, 500)
                    while wrong == ans:
                        wrong = random.randint(-50, 500)
                    ans = wrong
                examples.append(f"{format_val(a)} {target_symbol} {format_val(b)}{sep}{format_val(ans)}")
            
            while True:
                a = random.randint(*current_range)
                b = random.randint(*current_range)
                if not self.allow_repeats and (a, b) in used_pairs:
                    continue
                break
                
            query = f"{format_val(a)} {target_symbol} {format_val(b)}{sep}"
            answer = format_val(op_fn(a, b)) + "\n"
        elif self.task == "mapping":
            a = random.randint(*self.mapping_a_range)
            b = random.randint(*self.mapping_b_range)
            
            current_range = self.mapping_range
            if self.is_ood:
                if self.rule_family_idx is not None:
                    family_idx = self.rule_family_idx
                    fn = lambda x: self.rule_families[family_idx](x, a, b)
                elif self.mapping_ood_type == 'exponential':
                    # OOD Exponential Task Shift: y = x^a
                    a = random.randint(1, 4)
                    max_x = int(1000 ** (1/a))
                    current_range = (1, max_x)
                    fn = lambda x: x ** a
                else:
                    # OOD Extrapolation: Keep linear rule, but move x outside training range
                    current_range = (self.mapping_range[1] + 1, self.mapping_range[1] + 31)
                    fn = lambda x: a * x + b
            else:
                if self.rule_diversity:
                    if self.rule_family_idx is not None:
                        family_idx = self.rule_family_idx
                    elif self.exclude_family_idx is not None:
                        # Pick any family EXCEPT the excluded one
                        available = [i for i in range(len(self.rule_families)) if i != self.exclude_family_idx]
                        family_idx = random.choice(available)
                        
                    else:
                        family_idx = random.randint(0, len(self.rule_families) - 1)
                    fn = lambda x: self.rule_families[family_idx](x, a, b)
                else:
                    fn = lambda x: a * x + b
            
            # Generate context examples adhering to this function
            used_x = set()
            examples = []
            
            # Infinite Loop Safeguard: Ensure range has at least n_context + 2 unique values
            range_size = current_range[1] - current_range[0] + 1
            if range_size < current_n_context + 1:
                current_range = (current_range[0], current_range[0] + current_n_context + 2)

            while len(examples) < current_n_context:
                x = random.randint(*current_range)
                if x in used_x:
                    continue
                y = fn(x)
                if random.random() < self.noise_ratio:
                    y = random.randint(0, 100000)
                used_x.add(x)
                examples.append(f"{x}{sep}{y}")

            # Generate query that is not in the examples
            max_tries = 1000
            tries = 0
            while tries < max_tries:
                x = random.randint(*current_range)
                if x not in used_x:
                    break
                tries += 1
            
            if tries >= max_tries:
                # Fallback: find any valid x sequentially if random sampling fails
                for x in range(current_range[0], current_range[1] + 1):
                    if x not in used_x:
                        break
            
            y = fn(x)
            query = f"{x}{sep}"
            answer = str(y) + "\n"
        elif self.task == "decoding":
            # Rule: y = x (copy)
            # Use random tokens for decoding
            if self.is_ood:
                if self.decoding_reversal:
                    # Soft OOD: Same alphabet, but reversed relation
                    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                else:
                    # Hard OOD: Different alphabet (symbols and digits)
                    chars = list("!@#$%^&*()1234567890")
            else:
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
                
                # Query logic
                if self.is_ood and self.decoding_reversal:
                    # Relation Inversion: Context is k -> v, Query asks for v -> k
                    query_k = random.choice(seq_keys)
                    v_query = cipher_map[query_k]
                    query = f"{v_query} -> "
                    answer = query_k + "\n"
                else:
                    # Standard: Context is k -> v, Query asks for k -> v
                    query_k = random.choice(seq_keys)
                    query = f"{query_k} -> "
                    answer = cipher_map[query_k] + "\n"
            else:
                query_k = random.choice(chars)
                query = f"{query_k} -> "
                answer = cipher_map[query_k] + "\n"
        else:
            raise ValueError(f"Invalid task: {self.task}")
        prompt = "\n".join(examples) + "\n" + query
        if return_answer:
            answer = {"prompt": prompt, "answer": answer, }
            if self.task == "mapping" and 'family_idx' in locals():
                answer["rule_family_idx"] = family_idx
            return answer
        else:
            return prompt

    def build_dataset(self, return_answer=False):
        data = []
        while len(data) < self.n_samples:
            seq = self.generate_sequence(return_answer=return_answer)
            if seq is not None:
                data.append(seq)
        return data
