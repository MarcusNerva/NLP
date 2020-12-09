from models import BiLSTMClassifier

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from cfgs import get_total_settings
    args = get_total_settings()
    model = BiLSTMClassifier(args)
    text = '北京时间12月9日凌晨，欧冠第6轮小组赛中，巴萨主场0-3负于尤文图斯，尤文获得头名出线，巴萨小组第2，欧冠主场不败金身告破。\
    C罗罚中自己创造的点球，欧冠生涯首次攻破巴萨球门，下半时点球梅开二度，近13场做客诺坎普斩获14球，\
    麦肯尼凌空抽射破门，连续两场建功，博努奇进球被判无效。'
    print(model.classify(text))