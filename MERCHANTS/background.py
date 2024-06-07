from path import Path

# while True:
#     pass

while True:
    for i in range(100000000000000000000000):
        if i%1000000000 == 0:
            print(i)
            Path("/opt/amazon/sagemaker/sagemaker-code-editor-server-data/data/User/History/reset_timer.txt").touch()