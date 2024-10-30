import copy
import numpy as np
from collections import deque
from random import randrange
import datetime


# OPERATIONS DICTIONARY
# GLOBAL VARIABLES

def power(left, right):
    return left ** right

def product(left, right):
    return left * right

def add(left, right):
    return left + right

def subtract(left, right):
    return left - right

def divide(left, right): #TODO: MAKE SURE TO CHECK FOR DIVIDE BY ZERO CHECKS SOMEWHERE!!
    return left / right

def ln(left, right = -1):
    return np.log(left)

def exp(left, right = -1):
    return np.exp(left)

def sin(left, right = -1):
    return np.sin(left)

def cos(left, right = -1):
    return np.cos(left)

def nested_exponent(e):
    
    d = deque([e])
    
    while len(d) > 0:
        check_kids = True
        cur = d.pop()
        
        if cur.operation == "^":
            
            s = deque([cur.left, cur.right])
            check_kids = False
            
            while len(s) > 0:
                cur2 = s.pop()
                if cur2.operation != "const" and cur2.operation != "var":
                    s.appendleft(cur2.left)
                    s.appendleft(cur2.right)
                if cur2.operation == "^":
                    return True
        
        if check_kids and cur.operation != "const" and cur.operation != "var":
            d.appendleft(cur.left)
            d.appendleft(cur.right)
    
    return False


OPERATIONS_DICT = {"^": power,
                   "*": product,
                   "+": add,
                   "-": subtract,
                   "/": divide}#,"ln": ln,"exp": exp,"sin": sin,"cos": cos}

OPERATIONS_INPUTS = {2: ["^", "*", "+", "-", "/"],
                     1: ["ln", "exp", "sin", "cos"]}

def simplify(input_e):
    
    e = copy.deepcopy(input_e)
    
    if e.operation == "var" or e.operation == "const":
        return e
    
    e.left = simplify(e.left)
    e.left.par = e
    
    # now we are assuming left and right are simplified
    
    # CONSTANT CHECK... E.G. LN(C1) = C2, SIN(D1) = D2, etc.
    
    if e.right is None:
        if e.left.operation == "const":
            new = Expression("const", OPERATIONS_DICT[e.operation](e.left.left))
            return new
        return e
    
    e.right = simplify(e.right)
    e.right.par = e
        
    # SIMPLIFY CONSTANTS
    
    if e.left.operation == "const" and e.right.operation == "const":
        new = Expression("const", OPERATIONS_DICT[e.operation](e.left.left, e.right.left))
        return new
    
    # ASSOCIATIVE PROPERTY
    
    if e.operation == "+":
        if e.left.operation == "+" and e.right.operation == "const":
            dummy = copy.deepcopy(e.right)
            e.right = e.left
            e.right.par = e
            e.left = dummy
            e.left.par = e
            del dummy
            
        if e.left.operation == "const" and e.right.operation == "+":
            if e.right.left.operation == "const":
                e.left = Expression("const", e.left.left + e.right.left.left)
                e.left.par = e
                e.right = e.right.right
                e.right.par = e
            elif e.right.right.operation == "const":
                e.left = Expression("const", e.left.left + e.right.right.left)
                e.left.par = e
                e.right = e.right.left
                e.right.par = e
    
    if e.operation == "*":
        if e.left.operation == "*" and e.right.operation == "const":
            dummy = copy.deepcopy(e.right)
            e.right = e.left
            e.right.par = e
            e.left = dummy
            e.left.par = e
            del dummy
            
        if e.left.operation == "const" and e.right.operation == "*":
            if e.right.left.operation == "const":
                e.left = Expression("const", e.left.left * e.right.left.left)
                e.left.par = e
                e.right = e.right.right
                e.right.par = e
            elif e.right.right.operation == "const":
                e.left = Expression("const", e.left.left * e.right.right.left)
                e.left.par = e
                e.right = e.right.left
                e.right.par = e
    
    # DISTRIBUTIVE PROPERTY
    
    if e.operation == "*":
        if (e.left.operation == "+" or e.left.operation == "-") and e.right.operation == "const":
            dummy = copy.deepcopy(e.right)
            e.right = e.left
            e.right.par = e
            e.left = dummy
            e.left.par = e
            del dummy
        
        if e.left.operation == "const" and (e.right.operation == "+" or e.right.operation == "-"):
            
            e.operation = e.right.operation
            
            new_const = copy.deepcopy(e.left)
            
            e.left = Expression("*", e.left, e.right.left)
            e.left.par = e
            e.right = Expression("*", new_const, e.right.right)
            e.right.par = e

    e.update()
    return e

def random_expression(rand_round, input_vars):
    # input_vars = ["a", "b", "c"]
    
    l = len(input_vars)
    
    if rand_round == 0:
        if np.random.rand() <= 0.5:
            return Expression("const", np.random.rand())
        else:
            return Expression("var", np.random.choice(input_vars))
    
    elif rand_round == 1:
        if np.random.rand() <= 0: # 0.25 = CHANCE OF TYPE LN,EXP,SIN, OR COS..... HARD-CODED
            o = np.random.choice(OPERATIONS_INPUTS[1])
            v = np.random.choice(input_vars)
            left = Expression("var", v)
            right = None
        else:
            o = np.random.choice(OPERATIONS_INPUTS[2])
            
            c = np.random.rand() #  WHAT TO RANDOMLY SET CONSTANTS TO??
            v = np.random.choice(input_vars)
            
            if o == "^" and np.random.rand() <= 0.5:
                e = Expression("*", Expression("const", c), Expression("^", Expression("var", v),Expression("const", np.random.rand())))
                return e
                
            if l > 1:
                probs = [0.33, 0.33, 0.34]
            else:
                probs = [0.5, 0.5, 0]
            
            if o == "^":
                probs[1] *= 0.3
                probs[2] *= 0.3
                probs[0] = 1 - (probs[1] + probs[2])
            
            case = np.random.choice(["cv", "vc", "vv"], p=probs)
            
            if case == "cv":
                left = Expression("const", c)
                right = Expression("var", v)
            elif case == "vc":
                left = Expression("var", v)
                right = Expression("const", c)
            else:
                v1, v2 = np.random.choice(input_vars, size = 2, replace = False)
                left = Expression("var", v1)
                right = Expression("var", v2)
        
        e = Expression(o, left, right)
    else:
        
        while True:
            e = random_expression(rand_round - 1, input_vars) # depth is misleading... complexity instead maybe ?
            node_to_swap = e.select_random_node(select_type = "c/v")
            parent = node_to_swap.par
            new_sub_node = random_expression(1, input_vars)
            new_sub_node.par = parent

            if parent.left == node_to_swap:
                parent.left = new_sub_node
            elif parent.right == node_to_swap:
                parent.right = new_sub_node
            
            if not nested_exponent(e):
                break
                
    e.update()        
    return e
    
def delete_subtree(input_e, input_vars):
        
    parent = None
    while parent is None:
        if input_e.complexity > 3:
            node_to_delete = input_e.select_random_node("op")
        else:
            node_to_delete = input_e.select_random_node("any")
        parent = node_to_delete.par
    
    new_node = random_expression(0, input_vars)
    new_node.par = parent
    
    if parent.left == node_to_delete:
        parent.left = new_node
    elif parent.right == node_to_delete:
        parent.right = new_node
    
    input_e.update()
    

def mutate_expression_form(input_e, input_vars):
    
    while True:
        e = copy.deepcopy(input_e)
        while True:
            node_to_swap = e.select_random_node("any")
            parent = node_to_swap.par 
            if parent is not None:
                break
                
        node_to_swap_depth = node_to_swap.get_depth()

        if parent.left == node_to_swap:
            parent.left = random_expression(node_to_swap_depth, input_vars)
            
        elif parent.right == node_to_swap:
            parent.left = random_expression(node_to_swap_depth, input_vars)
            
        if not nested_exponent(e):
            break
    
    e.update()
    return e
    

class Expression():
    
    def __init__(self, operation, left, right = None):
        
        self.operation = operation # str: "var", "const", or operation from OPERATIONS_DICT
        self.par       = None
        self.left      = left      # int OR str OR Expression
        self.right     = right     # int OR str OR None OR Expression
        self.init_time = datetime.datetime.now()
        
        if self.operation != "const" and self.operation != "var":
            self.left.par = self
        if self.right is not None:
            self.right.par = self
        
        
        self.complexity = self.get_complexity()
        self.depth      = self.get_depth()
    
    def string_representation(self):
        
        if self.operation == "var":
            return self.left
        elif self.operation == "const":
            return str(self.left)
        else:
            left_representation = self.left.string_representation()
        
        if self.right is None:
            return self.operation + "(" + left_representation + ")"
        
        right_representation = self.right.string_representation()
        
        return "(" + left_representation + " " + self.operation + " " + right_representation + ")"
        
    def __hash__(self): 
        return hash(self.string_representation())
    
    def __eq__(self, other):
        if type(other) is type(self):
            return self.string_representation() == other.string_representation()
        else:
            return NotImplemented
    
    def update(self):
        
        # use whenever an already created expression has been modified
        # e.g. a lower node has been updated
        
        # TODO: UPDATE EVRYTHING BELOW THIS NODE
        
        self.get_depth()
        self.get_complexity()
    
    def get_depth(self):
        
        if self.operation == "var" or self.operation == "const":
            self.depth = 1
            return 1
        
        left_depth = self.left.get_depth()
        
        if self.right is None:
            right_depth = 0
        else:
            right_depth = self.right.get_depth()
        
        lower_depth = max(left_depth, right_depth)
        depth       = lower_depth + 1
        
        self.depth = depth
        
        return depth
        
    def get_complexity(self):
        #FIX THIS
        if self.operation == "var" or self.operation == "const":
            self.complexity = 1
            return 1
        
        left_complexity = self.left.get_complexity()
        
        if self.right is None:
            right_complexity = 0
        else:
            right_complexity = self.right.get_complexity()
        
        complexity = right_complexity + left_complexity + 1
        # + 1 for the operation node connecting the left to right
        
        self.complexity = complexity
        
        return complexity
    
    def mutate_random_constant(self, T):
        node = self.select_random_node("const")
        
        if node is None:
            return
        
        factor = (1 + T)**(2 * np.random.rand() - 1)
        
        if np.random.rand() < 0.5:
            factor *= -1
            
        node.left *= factor
        
        #self.update()
        
    def mutate_random_operator(self):
        node = self.select_random_node("op")
        node_op = node.operation
        
        while True:
            proposed_op = np.random.choice(list(OPERATIONS_DICT.keys()))
            
            if proposed_op != node_op and proposed_op != node.operation:
                node.operation = proposed_op
                if nested_exponent(node):
                    continue
                break
    
    def select_random_node(self, select_type = "any"):
        # select_type =      "any": select any node uniformly randomly
        # select_type =    "const": select any CONSTANT node uniformly randomly
        # select_type =      "var": select any VARIABLE node uniformly randomly
        # select_type =      "c/v": select any CONST/VAR node uniformly randomly
        # select_type =       "op": select any OPERATION node uniformly randomly
        
        d = deque([self])
        
        nodes_seen = 0
        selected = None
        
        while len(d) > 0:
            cur = d.pop()
            
            cur_type = cur.operation
            
            if select_type == "c/v" and (cur_type == "var" or cur_type == "const"):
                nodes_seen += 1

                if randrange(nodes_seen) == 0:
                    selected = cur
            
            elif select_type == "any" or cur_type == select_type:
                nodes_seen += 1
                
                if randrange(nodes_seen) == 0:
                    selected = cur
                    
            elif select_type == "op" and cur_type != "var" and cur_type != "const":
                nodes_seen += 1
                
                if randrange(nodes_seen) == 0:
                    selected = cur
                    
            if cur.operation != "const" and cur.operation != "var":
                if cur.right is not None:
                    d.append(cur.right)
                d.append(cur.left)
        
        return selected
    
    def compute(self, input_vars_dict):
        
        if self.operation == "const":
            return self.left
        elif self.operation == "var":
            return input_vars_dict[self.left]
        else:
            left_compute = self.left.compute(input_vars_dict)
        
        if self.right is None:
            right_compute = -1
        else:
            right_compute = self.right.compute(input_vars_dict)
        
        val = OPERATIONS_DICT[self.operation](left_compute, right_compute)
        
        return val