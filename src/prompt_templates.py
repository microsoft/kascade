"""
This logic is largely copied from the THUDM/LongBench
"""

def get_prompt_template(prompt_type, inst_tokens, use_inst_tokens=True):
    l_header = [
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n",  # hotpotqa, 2wikimqa, musique
        "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: ",  # narrativeqa
        "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: ",  # qasper
        "Read the following text and answer briefly.\n\n",  # multifieldqa_en
        "阅读以下文字并用中文简短回答：\n\n",  # multifieldqa_zh
        "请基于给定的文章回答下述问题。\n\n文章：",  # dureader
        "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n",  # gov_report
        "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n",  # qmsum
        "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n",  # multi_news
        "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n",  # vcsum
        "Please determine the type of the question below. Here are some examples of questions.\n\n",  # trec
        "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n",  # triviaqa
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n",  # samsum
        "请判断给定新闻的类别，下面是一些例子。\n\n",  # lsht
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n",  # passage_count
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n",  # passage_retrieval_en
        "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n",  # passage_retrieval_zh
        "Please complete the code given below. \n",  # lcc
        "Please complete the code given below. \n",  # repobench-p
        "Please reason step by step, and put your final answer within \\boxed{}.", # math500, aime
        "Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.", # math500, aime lighteval
    ]

    l_prompt_template_query = [
        "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {}\nAnswer:",  # hotpotqa, 2wikimqa, musique
        "\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {}\n\nAnswer:",  # narrativeqa
        "\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {}\n\nAnswer:",  # qasper
        "\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {}\nAnswer:",  # multifieldqa_en
        "\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{}\n回答：",  # multifieldqa_zh
        "\n\n请基于上述文章回答下面的问题。\n\n问题：{}\n回答：",  # dureader
        "\n\nNow, write a one-page summary of the report.\n\nSummary:",  # gov_report
        "\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {}\nAnswer:",  # qmsum
        "\n\nNow, write a one-page summary of all the news.\n\nSummary:",  # multi_news
        "\n会议总结：",  # vcsum
        "{}",  # trec
        "\n\n{}",  # triviaqa
        "\n\n{}",  # samsum
        "{}",  # lsht
        "\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",  # passage_count
        "\n\nThe following is an abstract.\n\n{}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",  # passage_retrieval_en
        "\n\n下面是一个摘要\n\n{}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",  # passage_retrieval_zh
        "Next line of code:\n",  # lcc
        "{}Next line of code:\n",  # repobench-p
        "\nProblem: {}\n", # math500, aime
        "\n\n{}",  # math500, aime lighteval
    ]


    if use_inst_tokens:
        header = ''.join([inst_tokens[0], l_header[prompt_type], inst_tokens[1]])
        query = ''.join([l_prompt_template_query[prompt_type], inst_tokens[2]])
        return header, query
    return l_header[prompt_type], l_prompt_template_query[prompt_type]