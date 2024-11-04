(define and
  (lambda
    (a b)
    (cond
      ((eq a #t) . (cond ((eq b #t) . #t) (t . #f)) )
      (t . #f)
    )
  )
)

(define take
  (lambda
    (n l)
    (cond
      ((eq n nil) . nil)
      (t .
        (cons
          (car l)
          (take (car n) (cdr l))
        )
      )
    )
  )
)

(define countdown
  (lambda
    (x m)
    (cond
      ((eq m nil) . x)
      ((eq x nil) . #f)
      (t . (countdown (car x) (car m)) )
    )
  )
)

(define divisible
  (lambda
    (x n)
    ((lambda
      (xx)
      (cond ((eq xx #f) . #f) ((eq xx nil) . #t) (t . (divisible xx n) ) )
    ) (countdown x n) )
  )
)

(define fb
  (lambda
    (i)
    (cond
      (((and ( (divisible i) (((nil))) )) ( (divisible i) (((((nil))))) )) . (quote fizzbuzz))
      (( (divisible i) (((nil))) ) . (quote fizz))
      (( (divisible i) (((((nil))))) ) . (quote buzz))
      (t . i)
    )
  )
)

(define fbfb
  (lambda
    (i)
    (cond
      ( ( and ( divisible i (((nil))) ) ( divisible i (((((nil))))) ) ) . (quote fizzbuzz))
      (( divisible i (((nil))) ) . (quote fizz))
      (( divisible i (((((nil))))) ) . (quote buzz))
      (t . i)
    )
  )
)

(define fizzesbuzzes
  (lambda
    (n)
    (cond
      ( (eq n (nil)) . ( (nil) ) )
      (t . ( (fb n) . (fizzesbuzzes (car n)) ))
    )
  )
)

(define fizzingbuzzing
  (lambda
    (i n)
    (cond
      ((eq i n) . ((fb i)))
      (t . ( (fb i) . (fizzingbuzzing (i) n) ))
    )
  )
)

(define fizzybuzzy
  (lambda
    (n)
    ( fizzingbuzzing (nil) n )
  )
)
