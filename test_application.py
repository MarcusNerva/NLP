from models import BiLSTMClassifier, TransformerClassifier, TextClassifierTransformer

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from cfgs import get_total_settings
    args = get_total_settings()
    # model = BiLSTMClassifier(args)
    # t_model = TransformerClassifier(args)
    t_model = TextClassifierTransformer(args)
    text = '张新军蔡启煌麓山并列领先 记分卡直播决赛轮新浪体育讯　\
    9月4日-7日，2008年欧米茄中巡赛麓山锦标赛将在成都麓山国际高尔夫俱乐部举行。\
    共有112人报名参加本站赛事，其中包括国家队7位队员在内的10位业余球员。'

    t_text = '''ApacheTomcat是美国阿帕奇（Apache）基金会的一款轻量级Web应用服务器。该程序实现了对Servlet和JavaServerPage（JSP）的支持。
    ApacheTomcat存在安全漏洞，该漏洞源于可以重用HTTP2连接上接收到的前一个流的HTTP请求头值，
    用于与后续流相关联的请求。虽然这很可能会导致错误和HTTP2连接的关闭，但信息可能会在请求之间泄漏。'''
    # print(model.classify(text))

    print(t_model.predict(t_text))