import Data.List (union)
data Expr = Var String | App Expr Expr | Abs String Expr

fresh_var :: String -> [String] -> String
fresh_var s xs = if s `elem` xs 
                   then fresh_var (s ++ "#") xs 
                   else s

free_vars :: Expr -> [String]
free_vars e = case e of 
  Var s -> [s]
  App m n -> (free_vars m) `union` (free_vars n)
  Abs x m -> filter (\x' -> x' /= x) (free_vars m)

subst :: Expr -> String -> Expr -> Expr
subst m x n = case m of
  Var s     -> if x == s then Var x else m
  App m' n' -> let m'' = subst m' x n 
                   n'' = subst n' x n 
                in App m'' n''
  Abs x' l  -> if x `elem` free_vars l
                 then Abs x' (subst l x n)
                 else Abs x' l
