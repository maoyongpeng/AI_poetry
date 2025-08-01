import dash
from dash import html
import feffery_antd_components as fac
from dash.dependencies import Input, Output,State
from sample import gen_model_ci,gen_model_shi
from dash import callback_context


app = dash.Dash(__name__)

app.layout = html.Div(
    id='main_div',
    children=[
    # 标题部分
    html.Div(
        id='title',
        children="AI诗词作家(Transformer)", style={
        "textAlign": "center",
        "fontSize": "80px",
        "fontFamily": "KaiTi",
         "color": "transparent",  # 使文本颜色透明，以便显示背景渐变色
        "background": "linear-gradient(orange, yellow, green, cyan, blue)",
        "background-clip": "text",
        "height": "15vh",  # 视口高度的百分比
        "width": "100%",
        "padding": "30px 0",  # 调整内边距以垂直居中文本
    }),
    html.Div(id='formessage',children=""),
    # 内容和边栏区
    fac.AntdRow([
        fac.AntdCol([
            # 增加文本框的字体大小和高度
            fac.AntdInput(
                          id='my_input',
                          placeholder="请输入符合生成方法的提示词字段!",mode='default',maxLength=1,
                          style={"marginBottom": "20px",
                                'fontSize': '25px',  # 字体大小
                                 "height": "70px",  # 高度
                                 'fontFamily': 'KaiTi',  # 设置字体为楷体
                                "width": "800px",
                                'backgroundColor':'rgba(255,255,255,0.5)',
                                'borderRadius': 20
                                }
                          ),# 宽度
            html.Br(),
            fac.AntdCard(
                title='诗词生成区',
                style={"width": "100%", "height": "70vh",
                       'backgroundColor':'rgba(255,255,255,0.6)',
                       'borderRadius': 20},
                children=html.Div(
                    # 使用一个Div将内容包裹起来，以便使用Flex布局
                    [
                        # 第一段文本
                        fac.AntdParagraph(id='paragraph1',
                        children="", code=True, copyable=True,style={"fontSize": "25px", "flex": "1",'fontFamily':"KaiTi"}),

                        # 分界线

                        fac.AntdDivider("分界", isDashed=True, fontColor="red", fontFamily="KaiTi", fontSize="20px"),

                        # 第二段文本
                        fac.AntdParagraph(id='paragraph2',
                        children="", code=True, copyable=True,style={"fontSize": "25px", "flex": "1",'fontFamily':"KaiTi"}),
                    ],
                    style={"display": "flex",
                           "flexDirection": "column",
                           "height": "100%"
                           }  # Flex布局，并确保容器高度充满卡片
                )
            ),

        ], span=18, style={"padding": "20px"}),  # 内容区占比约7

        fac.AntdCol([
            fac.AntdSpace(
                [
                    fac.AntdText("对象:", style={"fontSize": "24px",
                                                 "fontFamily": "KaiTi",
                                                 "color": "red",
                                                 'width':'100%'},
                                                 strong=True),
                    fac.AntdSelect(
                        id='my_select',
                        placeholder=f'请选择生成风格!',

                        options=[
                            {'label': '诗歌', 'value': '0'},
                            {'label': '词赋', 'value': '1'}
                        ],

                        status='warning',

                        style={
                             'width': 300
                        },
                        size='large',
                    ),
                ],
                style={"width": "100%", "height": "80%",'borderRadius': 20}
            ),
            fac.AntdSpace(
                [
                    fac.AntdText("方法:", style={"fontSize": "24px",
                                                 "fontFamily": "KaiTi",
                                                 "color": "blue",
                                                 'width': '100%'},
                                 strong=True),
                    fac.AntdSelect(
                        id='create_select',
                        placeholder=f'请选择生成方法!',
                        defaultValue='1',
                        options=[
                            {'label': '藏头类型生成', 'value': '0'}, # 选择时提示字段字数不少于4个汉字
                            {'label': '首字提示生成', 'value': '1'}, # 选择时提示字段字数只能是1个汉字
                            {'label': '首句提示生成', 'value': '2'}, # 选择时提示字段字数可以为5-8个汉字
                            {'label': '字词嵌入生成', 'value': '3'}  # 选择时提示字段字数可以为3-6个汉字
                        ],

                        status='warning',

                        style={
                            'width': 300
                        },
                        size='large',
                    ),
                ],
                style={"width": "100%", "height": "80%", 'borderRadius': 20}
            ),

            fac.AntdSpace([
            fac.AntdButton(
                            id='clear_button',
                            children="清空文本",block=True,type="primary",
                            style={"width": "100%", "height": "80px", "fontSize": "30px", "marginTop": "20px",
            "background": "linear-gradient(135deg, rgba(255, 99, 71, 0.8), rgba(255, 159, 64, 0.8), rgba(255, 218, 193, 0.8))",
                                'borderRadius': 20,}),
            html.Br(),
            html.Br(),
            fac.AntdButton(
                            id='generate_button',
                            children="生成",block=True,type="primary",
                            loadingChildren="生成中...",
                            autoSpin=True,
                            style={"width": "100%", "height": "80px", "fontSize": "30px", "marginTop": "20px",
            'background': 'linear-gradient(135deg, red, orange, yellow, green, cyan, blue, violet)',
                                  'borderRadius': 20})
            ],size="large",direction="vertical",style={"width": "100%", "height": "200px"}),

        ], span=6, style={"padding": "40px","height":"300px","width":"200px"}),  # 边栏区占比约3
    ], style={
        "height": "85vh",
    }),

], style={
    'backgroundImage': 'url("/assets/maitian.jpg")',
    'backgroundSize': 'cover',
    'backgroundRepeat': 'no-repeat', # 背景不重复
    'backgroundPosition': 'center', # 背景居中
    'backgroundAttachment': 'fixed', # 背景固定
    'height': '100vh',  # 视口高度
    'margin': '0'}
)



@app.callback(
    [Output('my_input', 'value'),  # 更新Input组件的value属性，使其内容被清空
     Output('my_select', 'value'), # 更新Select组件的value属性，使其选项被清空
     Output('create_select','value'), # 更新生成方法的value
     Output('formessage', 'children'),  # 更新显示消息的组件
     Output('paragraph1', 'children'),  # 更新第一个段落的内容
     Output('paragraph2', 'children'),
     Output('generate_button', 'loading')
     ],  # 更新第二个段落的内容
    [Input('clear_button', 'nClicks'),  # 监听两个按钮的点击
     Input('generate_button', 'nClicks')],
    [State('my_select', 'value'),  # 作为状态获取其他组件的值
     State('my_input', 'value'),
     State('create_select','value')],
    prevent_initial_call=True
)
def update_content(clear_clicks, generate_clicks, select_value, input_value,create_value):
    """
    更新内容区的回调函数
    :param clear_clicks:
    :param generate_clicks:
    :param select_value:
    :param input_value:
    :param create_value:
    :return:
    """
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'clear_button':
        # 如果是清空按钮触发的，清空输入和选择，更新提示消息
        return '', None, None,MessInfo('文本及选择已经清空','info'), '', '',False
    elif triggered_id == 'generate_button':
        # 生成逻辑
        if not select_value:
            message = "请选择一个生成类型!"
            return dash.no_update,dash.no_update, dash.no_update,MessInfo(message,'error'), dash.no_update, dash.no_update,False
        elif not create_value:
            message = '请选择一个生成方法!'
            return dash.no_update,dash.no_update, dash.no_update, MessInfo(message, 'error'), dash.no_update, dash.no_update, False
        elif not input_value:
            message = "请输入对应prompt提示词字段!"
            return dash.no_update,dash.no_update, dash.no_update,MessInfo(message,'error'), dash.no_update, dash.no_update,False


        elif not chinese(input_value):
            message = "包含非中文字段,请输入中文字符!"
            return dash.no_update,dash.no_update, dash.no_update,MessInfo(message,'error'), dash.no_update, dash.no_update,False
        if select_value and input_value and create_value:
            message = "生成成功!"
            if select_value=='1':
                if create_value=='0': #藏头
                    message = '藏头词赋' + message
                    text1 = gen_model_ci.gen_acrostic_T(start_words=input_value, temperature=0.7)
                    text2 = gen_model_ci.gen_acrostic_T(start_words=input_value, temperature=0.8)
                    return dash.no_update,dash.no_update, dash.no_update,MessInfo(message,'success'), text1, text2,False
                elif create_value=='1': #首字
                    message = '给定首字词赋' + message
                    text1 = gen_model_ci.generate(start_words=input_value, temperature=0.7)
                    text2 = gen_model_ci.generate(start_words=input_value, temperature=0.8)
                    return dash.no_update, dash.no_update, dash.no_update, MessInfo(message,'success'), text1, text2, False
                elif create_value=='2': # 首句
                    message = '给定首句词赋' + message
                    text1 = gen_model_ci.gen(start_words=input_value, temperature=0.7)
                    text2 = gen_model_ci.gen(start_words=input_value, temperature=0.8)
                    return dash.no_update, dash.no_update, dash.no_update, MessInfo(message,
                                                                                    'success'), text1, text2, False
                elif create_value=='3': #嵌入
                    message = '给定嵌入词' + message
                    text1 = gen_model_ci.generate_embed(embed_words=input_value, temperature=0.7)
                    text2 = gen_model_ci.generate_embed(embed_words=input_value, temperature=0.8)
                    return dash.no_update, dash.no_update, dash.no_update, MessInfo(message,
                                                                                    'success'), text1, text2, False
            elif select_value=='0':
                if create_value=='0':
                    message='藏头诗歌'+message
                    text1 = gen_model_shi.gen_acrostic_T(start_words=input_value, temperature=0.7)
                    text2 = gen_model_shi.gen_acrostic_T(start_words=input_value, temperature=0.8)
                    return dash.no_update,dash.no_update,dash.no_update, MessInfo(message, 'success'),text1, text2,False
                elif create_value=='1':
                    message = '给定首字诗歌' + message
                    text1 = gen_model_shi.generate(start_words=input_value, temperature=0.7)
                    text2 = gen_model_shi.generate(start_words=input_value, temperature=0.8)
                    return dash.no_update, dash.no_update, dash.no_update, MessInfo(message,
                                                                                    'success'), text1, text2, False
                elif create_value=='2':
                    message = '给定首句诗歌' + message
                    text1 = gen_model_shi.gen(start_words=input_value, temperature=0.7)
                    text2 = gen_model_shi.gen(start_words=input_value, temperature=0.8)
                    return dash.no_update, dash.no_update, dash.no_update, MessInfo(message,
                                                                                    'success'), text1, text2, False
                elif create_value=='3':
                    message = '给定嵌入词' + message
                    text1 = gen_model_shi.generate_embed(embed_words=input_value, temperature=0.7)
                    text2 = gen_model_shi.generate_embed(embed_words=input_value, temperature=0.8)
                    return dash.no_update, dash.no_update, dash.no_update, MessInfo(message,
                                                                                    'success'), text1, text2, False

    else:
        return dash.no_update, dash.no_update,dash.no_update, dash.no_update, dash.no_update, dash.no_update,False

@app.callback(
    Output('my_input', 'maxLength'),
    Output('my_input', 'placeholder'),
    Input('create_select', 'value'),
    prevent_initial_call=True
)
def update_input_limit(create_value):
    """
    根据生成方法动态限制输入框的字数
    :param create_value: 生成方法的值
    :return: maxLength, placeholder
    """
    if create_value == '0':  # 藏头类型生成
        return 4, "藏头生成,输入格式如: '深度学习','光风霁月'"
    elif create_value == '1':  # 首字提示生成
        return 1, "首字提示,输入格式如: '仙','草','夜'"
    elif create_value == '2':  # 首句提示生成
        return 8, "首句提示，输入格式如: '海内存知己','江山如此多娇'"
    elif create_value == '3':  # 字词嵌入生成
        return 6, "将字嵌入,输入格式如: '瑶桂香媚游','爱古蝶梦魂'"
    else:
        return 0, "请先选择生成方法，再输入符合生成方法的提示词字段!"


def MessInfo(text,typeinfo):
    return fac.AntdMessage(content=text,type=typeinfo,duration=3,maxCount=3)
def chinese(sentence):#判断是否为中文字符
    for s in sentence:
        if '\u4e00' <= s <= '\u9fff' or '\u3400' <= s <= '\u4dbf' or '\u20000' <= s <= '\u2a6df':
            continue
        else:
            return False
    return True

if __name__ == '__main__':
    app.run_server(debug=False)
    pass
