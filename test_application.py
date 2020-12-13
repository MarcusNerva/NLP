from models import TextClassifierLSTM, TextClassifierTransformer

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from cfgs import get_total_settings
    args = get_total_settings()
    model = TextClassifierLSTM(args)
    # t_model = TransformerClassifier(args)
    t_model = TextClassifierTransformer(args)
    text = '张新军蔡启煌麓山并列领先 记分卡直播决赛轮新浪体育讯　\
    9月4日-7日，2008年欧米茄中巡赛麓山锦标赛将在成都麓山国际高尔夫俱乐部举行。\
    共有112人报名参加本站赛事，其中包括国家队7位队员在内的10位业余球员。'

    text_1 = '''女星金晨身穿一袭金属绿色抹胸鱼尾长裙现身2020年国剧盛典红毯，
    这一次一起参加红毯活动的女明星很多，但金晨这次选择罕见的金属绿色更加让人眼前一亮，
    而且这条礼服在设计上也很凸显金晨高挑苗条的身材'''

    text_2 = '''是南京大屠杀纪念日83周年，众星在这一天为逝去的30万同胞发声，
    以缅怀这不能忘却的历史和伤痛。央视新闻于0点发起悼念，获得了大量公众人物的转发，
    目前已超10万网友评论，可见大家都对这一段历史仍铭记于心'''

    text_3 = '''ApacheTomcat是美国阿帕奇（Apache）基金会的一款轻量级Web应用服务器。该程序实现了对Servlet和JavaServerPage（JSP）的支持。
    ApacheTomcat存在安全漏洞，该漏洞源于可以重用HTTP2连接上接收到的前一个流的HTTP请求头值，
    用于与后续流相关联的请求。虽然这很可能会导致错误和HTTP2连接的关闭，但信息可能会在请求之间泄漏。'''
    # print(model.predict(text))

    text_4 = '''韩国2008年发生一起震惊社会的强奸案，
    犯人赵斗淳在厕所用极其残忍的手段强奸一个8岁女童，最后被判处12年有期徒刑。
    事件后来被翻拍成电影《素媛》，引起轩然大波，促使韩国修订未成年人保护法。'''


    print(t_model.predict(text_1))