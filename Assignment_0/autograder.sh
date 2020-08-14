echo "Grading Problem 1"

python3 probability.py --seed 42 --num 100 --lamda 1
python3 ref_probability.py --seed 42 --num 100 --lamda 1
DIFF=$(diff 'problem_1.txt' 'ref_problem_1.txt') 
if [ "$DIFF" != "" ] 
then
    echo "Failed TC 1"
else
    echo "Passed TC 1"
fi

python3 probability.py --seed 42 --num 100 --lamda 2
python3 ref_probability.py --seed 42 --num 100 --lamda 2
DIFF=$(diff 'problem_1.txt' 'ref_problem_1.txt') 
if [ "$DIFF" != "" ] 
then
    echo "Failed TC 2"
else
    echo "Passed TC 2"
fi

python3 probability.py --seed 50 --num 100 --lamda 1
python3 ref_probability.py --seed 50 --num 100 --lamda 1
DIFF=$(diff 'problem_1.txt' 'ref_problem_1.txt') 
if [ "$DIFF" != "" ] 
then
    echo "Failed TC 3"
else
    echo "Passed TC 3"
fi

python3 probability.py --seed 50 --num 100 --lamda 2
python3 ref_probability.py --seed 50 --num 100 --lamda 2
DIFF=$(diff 'problem_1.txt' 'ref_problem_1.txt') 
if [ "$DIFF" != "" ] 
then
    echo "Failed TC 4"
else
    echo "Passed TC 4"
fi



echo "------------GRADING PROBLEM 2------------"
python3 grade_2.py --num 100 
