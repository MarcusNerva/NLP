from models import BiLSTMClassifier

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from cfgs import get_total_settings
    args = get_total_settings()
    model = BiLSTMClassifier(args)
    text = '张新军蔡启煌麓山并列领先 记分卡直播决赛轮新浪体育讯　\
    9月4日-7日，2008年欧米茄中巡赛麓山锦标赛将在成都麓山国际高尔夫俱乐部举行。\
    共有112人报名参加本站赛事，其中包括国家队7位队员在内的10位业余球员。'
    print(model.classify(text))