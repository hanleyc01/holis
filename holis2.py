#! /usr/local/bin/python3.13
"""
holis2.py

holis2 is a reimplementation of holis, with added type annotations, as
well as a reworking of the memory system. 

It comes with two modes: the repl, and the script interpreter. In order
to start the repl, run the program as such:

$ ./holis2.py --repl
$ ./holis2.py -r

In order to interpret a LISP file,

$ ./holis2.py --interpret <filename>
$ ./holis2.py -i <filename>

Where <filename> is the name of the source-file.

To see this message again, simply run the file with no arguments,
or with the argument --help.

$ ./holis2.py --help
$ ./holis2.py -h

For verbose running of the interpreter,

$ ./holis2.py --verbose [rest of the args]
$ ./holis2.py -v [rest of the args]
"""

from __future__ import annotations
import math
from collections.abc import Sequence
import typing as t
import sys
from enum import Enum, auto
from dataclasses import dataclass
from pprint import pprint
from abc import ABC
from dataclasses import dataclass
from itertools import takewhile

import numpy as np
from numpy.fft import fft, ifft
import numpy.typing as npt

VERBOSE = False


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.arange(perm.size)
    inv[perm] = np.arange(perm.size)
    return inv


class HRR(Sequence):
    """
    Class for Tony Plate's Holographic Reduced Representations.
    """

    large: float
    small: float
    v: npt.NDArray
    scale: np.floating | npt.NDArray
    shape: t.Any
    size: int

    def __init__(
        self, data: npt.ArrayLike | HRR, large: float = 0.8, small: float = 0.2
    ) -> None:
        """
        Initialize a new HRR. In order to make an zero'd out array, use
        `HRR.zeros`, for a vector of values sampled from the normal
        distribution, with a mean of zero and a standard deviation of `1/N`,
        use `HRR.normal`.
        """
        self.large = large
        self.small = small
        self.v = np.array(data, dtype=float)
        self.scale = np.linalg.norm(self.v)
        self.shape = self.v.shape
        self.size = self.v.size

    @staticmethod
    def normal(N: int, large: float = 0.8, small: float = 0.2) -> HRR:
        """
        Create an HRR with a vector of length `N` using values sampled
        from a normal distribution with a mean of zero and a standard
        deviation of `1/N`.
        """
        sd = 1.0 / math.sqrt(N)
        v = np.random.normal(scale=sd, size=N)
        v /= np.linalg.norm(v)
        return HRR(v, large=large, small=small)

    @staticmethod
    def zeros(N: int, large: float = 0.8, small: float = 0.2) -> HRR:
        """
        Create an HRR with a vector of zeros of length `N`.
        """
        return HRR(np.zeros(N), small=small, large=large)

    def __array__(
        self, dtype: type[t.Any] | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")
        if dtype is None:
            dtype = float
        return np.array(self.v, dtype=dtype, copy=True)

    @t.overload
    def __getitem__(self, key: int, /) -> float:
        ...

    @t.overload
    def __getitem__(self, key: slice, /) -> HRR:
        ...

    def __getitem__(self, key, /) -> float | HRR:
        if isinstance(key, int):
            return self.v[key]
        else:
            return HRR(self.v[key])

    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        self.v[key] = value

    def __len__(self) -> int:
        return self.v.size

    def __mul__(self, other: HRR | npt.NDArray | float | int) -> HRR:
        """
        Multiplication-like operation which is used for binding or
        association.
        """
        if isinstance(other, HRR):
            return HRR(ifft(fft(self.v) * fft(other.v)).real)
        else:
            return HRR(self.v * other)

    # Type ignore justification:
    # It warns about "unsafe overlapping", however, in usage this is not so.
    def __rmul__(self, other: HRR | npt.NDArray | float | int) -> HRR:  # type: ignore
        return self * other

    def __pow__(self, exponent: float | npt.NDArray) -> HRR:
        """
        Fractional binding.
        """
        return HRR(ifft(fft(self.v).__pow__(exponent)).real)

    def __add__(self, other: HRR | HRRScale | npt.NDArray | float | int) -> HRR:
        """
        Addition-like operation that superposes vectors or adds them to the
        set.
        """
        if isinstance(other, HRR) or isinstance(other, HRRScale):
            return other + self.v
        else:
            return HRR(other + self.v)

    def __neg__(self) -> HRR:
        return HRR(-self.v)

    def __sub__(self, other: HRR | npt.ArrayLike) -> HRR:
        if isinstance(other, HRR):
            return HRR(self.v - other.v)
        else:
            return HRR(self.v - other)

    def __invert__(self) -> HRR:
        """
        Invert the HRR such that binding with the inverse unbinds.
        """
        return HRR(self.v[np.r_[0, self.v.size - 1 : 0 : -1]])

    def __truediv__(self, other: HRR | npt.ArrayLike) -> HRR:
        """
        Unbinding operation cancels out binding.
        """
        if isinstance(other, HRR):
            return self * ~other
        else:
            return HRR(self.v / other)

    def magnitude(self) -> float:
        return math.sqrt(self.v @ self.v)

    # Type ignore justification:
    # Our goal here is to deploy `eq` is a similarity rating in order to make
    # similarity computations easier on the eyes. This might violate the
    # global standards of python, but for this project we don't need to worry.
    def __eq__(self, other: HRR) -> float:  # type: ignore
        """
        Compare two vectors using vector cosine to measure similarity.
        """
        scale = self.scale * other.scale
        if scale == 0:
            return 0
        return (self.v @ other.v) / scale

    def unit(self) -> HRR:
        return HRR(self.v / self.scale)

    def __matmul__(self, other: HRR | npt.ArrayLike) -> HRR | npt.ArrayLike:
        if isinstance(other, HRR):
            return self.v @ other.v
        elif isinstance(other, np.ndarray) and len(other.shape) == 2:
            return HRR(self.v @ other)
        else:
            return self.v @ other

    def __or__(self, other: HRR) -> HRR:
        """
        Variant addition operator.
        """
        if self.scale > self.large:
            return self
        elif self.scale < self.small:
            return other
        else:
            return self + other

    def __repr__(self) -> str:
        return f"HRR({self.v})"

    def __key(self) -> t.Tuple[HRR]:
        return (*self.v,)

    def __hash__(self) -> int:
        return hash(self.__key())


class HRRScale:  #
    pass


class SimpleCleanup:
    """
    Simple cleanup memory.
    """

    M: npt.NDArray
    "Memory matrix"
    n: int
    "Trace dimensionality"
    m: int
    "Initial maximum of traces"
    k: int
    "Increment for adding more traces"
    i: int
    "Current number of traces"

    def __init__(self, n: int = 512, m: int = 100) -> None:
        self.M = np.zeros((m, n))
        self.n = n
        self.m = m
        self.k = m
        self.i = 0

    def memorize(self, activation: HRR) -> HRR:
        if self.i >= self.m:
            self.M = np.concatenate([self.M, np.zeros((self.k, self.n))], axis=0)
        self.M[self.i, :] = activation.v
        self.i += 1
        return activation

    def recall(self, probe: HRR) -> HRR:
        trace = self.M @ probe.v
        echo = HRR(self.M[np.argmax(trace), :])
        return echo


class Lexicon:
    map: dict[str, HRR]
    reverse: dict[HRR, str]

    def __init__(self) -> None:
        self.map = {}
        self.reverse = {}

    def __getitem__(self, key: str) -> HRR:
        return self.map[key]

    def __setitem__(self, key: str, value: HRR) -> None:
        self.map[key] = value
        self.reverse[value] = key

    def update(self, d: dict[str, HRR]) -> None:
        self.map.update(d)
        self.reverse.update(((v, k) for k, v in d.items()))

    def __list__(self) -> list[tuple[str, HRR]]:
        return list(self.map.items())


################################################################################
# Tokens                                                                       #
################################################################################


class TokenKind(Enum):
    "Enumeration of different tags for tagged union of tokens."
    Cond = auto()
    Define = auto()
    Dot = auto()
    Error = auto()
    Ident = auto()
    If = auto()
    Int = auto()
    Lambda = auto()
    Left_paren = auto()
    Right_paren = auto()


@dataclass
class Token:
    """
    Dataclass representing source-level tokens in the provided text. Includes
    a `kind` to function as a tag, an `offset` picking it out in the text,
    and finally, a `cont` which is the raw text content.
    """

    kind: TokenKind
    cont: str
    offset: int


class Lexer:
    """
    Scanner class which reads through the source text.
    """

    _offset: int
    _cont: str

    _reserved = {
        "cond": TokenKind.Cond,
        "define": TokenKind.Define,
        "if": TokenKind.If,
        "lambda": TokenKind.Lambda,
        ".": TokenKind.Dot
    }

    def __init__(self, offset: int, cont: str) -> None:
        self._cont = cont
        self._offset = offset

    def _peek(self, offset: int = 0) -> str | None:
        try:
            return self._cont[self._offset + offset]
        except IndexError:
            return None

    def _next(self) -> None:
        self._offset += 1

    def _poke(self, offset: int = 0) -> str | None:
        self._next()
        return self._peek(-1)

    def _token(self, kind: TokenKind, cont: str) -> Token:
        "Generate a new token from provided `TokenKind` and `cont`."
        return Token(kind, cont, offset=self._offset)

    def _error(self, msg: str) -> Token:
        "Make an error token with provided `msg` at `self._offset`."
        return Token(TokenKind.Error, cont=msg, offset=self._offset)

    def _skip_whitespace(self) -> None:
        curr = self._peek()
        while (curr is not None) and curr.isspace():
            self._next()
            curr = self._peek()

    @staticmethod
    def _issymbol(char: str | None) -> bool:
        "Check if character is a valid symbol."
        if char is None:
            return False
        return char.isalpha() or char in [
            "!",
            "#",
            "$",
            "%",
            "*",
            "-",
            "=",
            "?",
            "@",
            "^",
            ".",
            "<",
            ">",
        ]

    def _symbol_or_ident(self, x: str) -> tuple[TokenKind, str]:
        "Checks if string is in the reserved keywords, otherwise return an ident"
        if x.lower() in self._reserved:
            return self._reserved[x.lower()], x
        else:
            return TokenKind.Ident, x

    def token(self) -> Token | None:
        self._skip_whitespace()
        curr = self._poke()
        if curr is None:
            return None
        elif curr == "(":
            return self._token(TokenKind.Left_paren, curr)
        elif curr == ")":
            return self._token(TokenKind.Right_paren, curr)
        elif curr.isdigit():
            digit = [curr]

            next = self._peek()
            while next is not None and next.isdigit():
                digit.append(next)
                self._next()
                next = self._peek()

            return self._token(TokenKind.Int, "".join(digit))
        elif curr.isalpha() or self._issymbol(curr):
            symbol = [curr]

            next = self._peek()
            while next is not None and (next.isalpha() or self._issymbol(next)):
                symbol.append(next)
                self._next()
                next = self._peek()

            kind, cont = self._symbol_or_ident("".join(symbol))
            return self._token(kind, cont)
        else:
            return self._error(f"Unrecognized token, {curr}")


def report_error(err: Token, src: str,  src_name: str) -> None:
    """Report an error."""
    vprint(f"reporting error {err}")
    lin, col = 0, 0
    for char in src:
        if char == "\n":
            col += 1
            lin = 0
        else:
            lin += 1
    print(f"{src_name}:{lin}:{col} [Lexical error] {err.cont}",file=sys.stderr)


def tokenize(src: str, src_name: str) -> list[Token]:
    if len(src) == 0:
        return []

    lexer = Lexer(0, src)
    current_token = lexer.token()
    tokens: list[Token] = []
    while current_token is not None:
        tokens.append(current_token)
        current_token = lexer.token()

    errs = [error for error in tokens if error.kind == TokenKind.Error]
    if len(errs) == 0:
        return tokens
    else:
        for err in errs:
            report_error(err, src, src_name)
            exit(-1)

    return tokens


################################################################################
# Syntax                                                                       #
################################################################################


class lispexpr(ABC):
    """
    We define a `lispexpr` as being a syntactic representation of the source
    level text provided by the user.
    ```
    E_1, ..., E_n ::= x 
                    | c 
                    | <digit>
                    | (E_1 [E_2 ... E_n])
                    | (define <name> E_1)
                    | (define (name <arg1> [<arg2> ... <arg_n>]) E_1)
                    | (if E1 E2 E3)
                    | (cond [(<condition> . <result>)]+)

    <condition>  ::= E 
    <result>   ::= E
    <name> ::= x
    <arg_n> ::= x
    <digit> ::= [0-9]+
    ```
    """
    pass


@dataclass
class Var(lispexpr):
    x: str
    offset: int


@dataclass
class Constant(lispexpr):
    c: str
    offset: int


@dataclass
class Digit(lispexpr):
    i: int
    offset: int


@dataclass
class Lambda(lispexpr):
    args: list[str]
    body: lispexpr
    offset: int


@dataclass
class Application(lispexpr):
    rator: lispexpr
    rand: list[lispexpr]
    offset: int


@dataclass
class Define(lispexpr):
    name: str
    body: lispexpr
    offset: int


@dataclass
class If(lispexpr):
    condition: lispexpr
    then: lispexpr
    otherwise: lispexpr
    offset: int


@dataclass
class Cond(lispexpr):
    conds: list[tuple[lispexpr, lispexpr]]
    offset: int


@dataclass
class ParserError(Exception):
    msg: str
    offset: int


class Parser:
    _src: list[Token]
    _offset: int

    def __init__(self, src: list[Token], offset: int) -> None:
        self._src = src
        self._offset = offset

    def _peek(self, offset: int = 0) -> Token | None:
        #vprint(f"peek at: {self._offset}, {self._src[self._offset]}")
        try:
            return self._src[self._offset + offset]
        except IndexError:
            return None

    def _next(self) -> None:
        self._offset += 1

    def _poke(self, offset: int = 0) -> Token | None:
        self._next()
        return self._peek(offset=-1)

    def _check_next(self, kind: TokenKind, msg: str, offset: int) -> None:
        curr = self._poke()
        if curr is None or curr.kind != kind:
            raise ParserError(f"{msg}, {curr}", offset)

    def _is_special_form(self, kind: TokenKind) -> bool:hah
        return kind in [TokenKind.Define, TokenKind.Lambda, TokenKind.Cond, TokenKind.If]

    def _parse_special_form(self, special: Token) -> lispexpr | None:
        if special.kind == TokenKind.Define:
            name_token = self._poke()
            name: str | None = None
            if name_token is None or name_token.kind != TokenKind.Ident:
                raise ParserError("Error: in `define` context expected identifier", special.offset)
            elif name_token.kind == TokenKind.Ident:
                name = name_token.cont
            body = self.parse_expr(nested=True)
            if body is None:
                body = Var("#undefined", name_token.offset)
            self._check_next(TokenKind.Right_paren, "Error: in `define` context, expected closing right parenthesis after expression", body.offset) # type:ignore
            return Define(name, body, name_token.offset)
        elif special.kind == TokenKind.Lambda:
            self._check_next(TokenKind.Left_paren, "Error: in `lambda` context, expected opening `(` for argument list, got", special.offset)
            def is_ident_and_move_ptr(x: Token) -> bool:
                if x.kind == TokenKind.Ident:
                    self._next()
                    return True
                else:
                    return False

            args = [arg.cont for arg in takewhile(is_ident_and_move_ptr, self._src)]
            self._check_next(TokenKind.Right_paren, "Error: in `lambda` context, expected closing ')' for argument list, got", special.offset)
            body = self.parse_expr(nexted=True)
            return Lambda(args, body)
        elif special.kind == TokenKind.Cond:
            return None #remove when finished
        elif special.kind == TokenKind.If:
            return None #remove when finished
        else: 
            return None
        
    def _parse_application_or_special_form(self) -> lispexpr | None:
        "Parse a `lispexpr` which follows the form `(..)`."
        curr = self._poke()
        if curr is None:
            raise ParserError("Error: unclosed parenthesis", self._offset)

        if self._is_special_form(curr.kind):
            return self._parse_special_form(curr)
        else:
            rands = []
            rand = self.parse_expr(nested=True)
            while rand is not None:
                rands.append(rand)
                rand = self.parse_expr(nested=True)
        
            return Application(curr, rand=rands, offset=curr.offset) #type: ignore
        
    def parse_expr(self, nested: bool = False) -> lispexpr | None:
        "Parse a `lispexpr`."
        curr = self._poke()
        if curr is None:
            return None
        
        if curr.kind == TokenKind.Left_paren:
            return self._parse_application_or_special_form()
        elif curr.kind == TokenKind.Int:
            return Digit(int(curr.cont), curr.offset)
        elif curr.kind == TokenKind.Ident:
            return Var(curr.cont, curr.offset)
        elif curr.kind == TokenKind.Right_paren and not nested:
            raise ParserError("Error: unexpected right paren", curr.offset)
        elif curr.kind == TokenKind.Right_paren and nested:
            return None
        elif curr.kind == TokenKind.Left_paren:
            return self._parse_application_or_special_form()
        elif curr.kind == TokenKind.Error:
            raise ParserError("This label is the target of a goto from outside of the block containing this label AND this block has an automatic variable with an initializer AND your window wasn't wide enough to read this whole error message", curr.offset)
        else:
            raise ParserError(f"Error: unexpected {curr.cont}", curr.offset)


def parse(src: str, src_name: str) -> list[lispexpr]:
    vprint("Parsing")
    tokens = tokenize(src, src_name)
    if VERBOSE:
        print("Tokenization results: ", end="")
        pprint(tokens)

    if len(tokens) == 0:
        return []

    parser = Parser(tokens, 0)
    exprs = []
    curr = parser.parse_expr()
    while curr is not None:
        exprs.append(curr)
        curr = parser.parse_expr()
    return exprs


################################################################################
# Main                                                                         #
################################################################################

interpret_error_str = """
Please provide a source-file for interpretation following the format

$ ./holis2.py --interpret <filename>
$ ./holis2.py -i <filename>
"""


def vprint(*args) -> None:
    """
    Print if VERBOSE == True
    """
    if VERBOSE:
        print(*args)


def run(src: str) -> None:
    vprint(f"entering into run, with argument {src}")
    pass


def repl() -> None:
    """
    Read-Eval-Print loop.
    """
    vprint("entering into repl")
    prompt = "holis2.py> "
    while True:
        txt = input(prompt)
        if txt.strip() in ["", "(exit)", "(quit)", "exit", "quit"]:
            print("Goodbye!")
            exit(0)
        run(txt)


def interpret(src: str) -> None:
    """
    Interpret file at `src`.
    """
    vprint("interpreting %s" % src)
    with open(src, "r") as f:
        contents = f.read()
    parsed_program = parse(contents, src_name=src)
    print(parsed_program)


def main() -> None:
    if len(sys.argv) == 1:
        print(__doc__)

    cmds = sys.argv[1:]

    # Check to see if verbose is true
    if "-v" in cmds or "--verbose" in cmds:
        global VERBOSE
        VERBOSE = True
    vprint(f"arguments: {cmds}")
    cmds = [cmd for cmd in cmds if cmd != "-v" or cmd != "--verbose"]

    # iterate throught the commands and respond accordingly
    for i, cmd in enumerate(cmds):
        if cmd == "--help" or cmd == "-h":
            print(__doc__)
            return
        elif cmd == "--repl" or cmd == "-r":
            repl()
        elif cmd == "--interpret" or cmd == "-i":
            try:
                in_file = cmds[i + 1]
            except IndexError:
                print(interpret_error_str)
                exit(-1)
            interpret(in_file)



if __name__ == "__main__":
    main()
