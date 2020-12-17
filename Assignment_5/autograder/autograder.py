from grade_layers import grade_layers
from grade_applications import grade_applications
from grade_trainer_train import grade_trainer_train

marks = grade_layers() + grade_trainer_train() + grade_applications()

print('Total Marks = ', marks)
