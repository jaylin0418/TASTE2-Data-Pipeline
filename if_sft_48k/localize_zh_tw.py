"""
Replace a subset of examples in dataset.jsonl with Taiwan-localized content.
Run repeatedly safe: always restores from .bak first, then applies full TW_POOL.
"""
import json, random, shutil
from collections import defaultdict
from pathlib import Path

INPUT  = "/work/jaylin0418/IF_data_generation/output_zh/full/dataset.jsonl"
BACKUP = "/work/jaylin0418/IF_data_generation/output_zh/full/dataset.jsonl.bak"

# Format: (style, ability, instruction, target_text)
TW_POOL = [

    # ══════════════════════════════════════════════
    # read_aloud
    # ══════════════════════════════════════════════
    ("none","instruction_following/read_aloud","請說這句話：台北的夜市熱鬧非凡。","台北的夜市熱鬧非凡。"),
    ("none","instruction_following/read_aloud","請唸出這句話：珍珠奶茶是台灣最受歡迎的飲料之一。","珍珠奶茶是台灣最受歡迎的飲料之一。"),
    ("none","instruction_following/read_aloud","跟著我說：今天搭捷運上班，車廂裡擠滿了人。","今天搭捷運上班，車廂裡擠滿了人。"),
    ("none","instruction_following/read_aloud","請說：媽祖遶境是台灣重要的民俗活動。","媽祖遶境是台灣重要的民俗活動。"),
    ("none","instruction_following/read_aloud","請唸：阿里山的雲海每天清晨都特別美麗。","阿里山的雲海每天清晨都特別美麗。"),
    ("none","instruction_following/read_aloud","請說這句話：九份的老街充滿了懷舊氣氛。","九份的老街充滿了懷舊氣氛。"),
    ("none","instruction_following/read_aloud","請唸出這句話：台灣的高鐵把南北的距離縮短了很多。","台灣的高鐵把南北的距離縮短了很多。"),
    ("none","instruction_following/read_aloud","跟著我說：每年元宵節，台灣各地都會舉辦燈會。","每年元宵節，台灣各地都會舉辦燈會。"),
    ("none","instruction_following/read_aloud","請說：鹽水蜂炮是台南著名的元宵節傳統活動。","鹽水蜂炮是台南著名的元宵節傳統活動。"),
    ("none","instruction_following/read_aloud","請唸：誠品書店二十四小時不打烊，是很多人的深夜好去處。","誠品書店二十四小時不打烊，是很多人的深夜好去處。"),
    ("none","instruction_following/read_aloud","請說這句話：烏來的溫泉在冬天特別受歡迎。","烏來的溫泉在冬天特別受歡迎。"),
    ("none","instruction_following/read_aloud","請唸：台灣每年夏天都要面對好幾個颱風的威脅。","台灣每年夏天都要面對好幾個颱風的威脅。"),
    ("none","instruction_following/read_aloud","跟著我說：基隆廟口夜市的小吃種類多到讓人眼花撩亂。","基隆廟口夜市的小吃種類多到讓人眼花撩亂。"),
    ("none","instruction_following/read_aloud","請說：台灣的便利商店幾乎二十四小時都可以繳費、取件。","台灣的便利商店幾乎二十四小時都可以繳費、取件。"),
    ("none","instruction_following/read_aloud","請唸出：花蓮的太魯閣峽谷是台灣最壯麗的自然景觀之一。","花蓮的太魯閣峽谷是台灣最壯麗的自然景觀之一。"),
    ("none","instruction_following/read_aloud","請說這句話：台南的赤崁樓見證了台灣數百年的歷史變遷。","台南的赤崁樓見證了台灣數百年的歷史變遷。"),
    ("none","instruction_following/read_aloud","請唸：屏東的墾丁每年都吸引大批遊客前往衝浪。","屏東的墾丁每年都吸引大批遊客前往衝浪。"),
    ("none","instruction_following/read_aloud","跟著我說：台北的公共腳踏車叫做YouBike，非常方便。","台北的公共腳踏車叫做YouBike，非常方便。"),
    ("none","instruction_following/read_aloud","請說：台灣的健保制度讓民眾可以用合理的費用看診。","台灣的健保制度讓民眾可以用合理的費用看診。"),
    ("none","instruction_following/read_aloud","請唸出這句話：淡水的漁人碼頭在夕陽西下時美得像一幅畫。","淡水的漁人碼頭在夕陽西下時美得像一幅畫。"),
    ("none","instruction_following/read_aloud","請說這句話：台東的池上稻田每到秋收季節就變成一片金黃。","台東的池上稻田每到秋收季節就變成一片金黃。"),
    ("none","instruction_following/read_aloud","請唸：宜蘭的冬山河親水公園是許多親子假日的好去處。","宜蘭的冬山河親水公園是許多親子假日的好去處。"),
    ("none","instruction_following/read_aloud","跟著我說：台灣的廟宇建築精緻，細節雕工令人嘆為觀止。","台灣的廟宇建築精緻，細節雕工令人嘆為觀止。"),
    ("none","instruction_following/read_aloud","請說：平溪放天燈是台灣最浪漫的新年習俗之一。","平溪放天燈是台灣最浪漫的新年習俗之一。"),
    ("none","instruction_following/read_aloud","請唸出：台灣的棒球文化深植人心，每逢國際賽事全民瘋棒球。","台灣的棒球文化深植人心，每逢國際賽事全民瘋棒球。"),
    ("none","instruction_following/read_aloud","請說這句話：彰化的鹿港老街保留了清朝時期的閩南建築風格。","彰化的鹿港老街保留了清朝時期的閩南建築風格。"),
    ("none","instruction_following/read_aloud","請唸：台北故宮博物院收藏了數十萬件珍貴的中華文物。","台北故宮博物院收藏了數十萬件珍貴的中華文物。"),
    ("none","instruction_following/read_aloud","跟著我說：苗栗的客家文化節每年都吸引大批遊客參加。","苗栗的客家文化節每年都吸引大批遊客參加。"),
    ("none","instruction_following/read_aloud","請說：台灣的夜市文化已經成為外國觀光客必訪的體驗。","台灣的夜市文化已經成為外國觀光客必訪的體驗。"),
    ("none","instruction_following/read_aloud","請唸出這句話：澎湖的海水清澈見底，珊瑚礁多彩繽紛。","澎湖的海水清澈見底，珊瑚礁多彩繽紛。"),

    # ══════════════════════════════════════════════
    # listing
    # ══════════════════════════════════════════════
    ("none","instruction_following/listing","說出三種台灣著名小吃。","蚵仔煎、鹽酥雞和滷肉飯。"),
    ("none","instruction_following/listing","列出三個台灣著名夜市。","士林夜市、饒河夜市和六合夜市。"),
    ("none","instruction_following/listing","說出三個台灣城市。","台北、台中和高雄。"),
    ("none","instruction_following/listing","列出兩個台灣著名自然景點。","阿里山和日月潭。"),
    ("none","instruction_following/listing","說出四種台灣常見的手搖飲料。","珍珠奶茶、仙草奶茶、芋頭牛奶和冬瓜茶。"),
    ("none","instruction_following/listing","說出三種台灣傳統節慶。","元宵節、端午節和中秋節。"),
    ("none","instruction_following/listing","列出兩種台灣的大眾交通工具。","捷運和高鐵。"),
    ("none","instruction_following/listing","說出三種台灣常見的路邊攤食物。","臭豆腐、蚵仔麵線和大腸包小腸。"),
    ("none","instruction_following/listing","列出三個台灣知名大學。","台灣大學、清華大學和成功大學。"),
    ("none","instruction_following/listing","說出兩種台灣特有的水果。","釋迦和蓮霧。"),
    ("none","instruction_following/listing","列出三種台灣傳統糕點。","鳳梨酥、太陽餅和牛軋糖。"),
    ("none","instruction_following/listing","說出兩個台灣知名科技公司。","台積電和鴻海。"),
    ("none","instruction_following/listing","列出三種台灣的傳統民俗活動。","廟會、陣頭和放天燈。"),
    ("none","instruction_following/listing","說出三個台北著名景點。","台北一○一、故宮博物院和西門町。"),
    ("none","instruction_following/listing","列出三種台灣常見的早餐食物。","蛋餅、燒餅油條和飯糰。"),
    ("none","instruction_following/listing","說出四個台灣的縣市。","台北、新北、台中和高雄。"),
    ("none","instruction_following/listing","列出三種台灣常見的伴手禮。","鳳梨酥、牛軋糖和土鳳梨酥。"),
    ("none","instruction_following/listing","說出三個台灣的離島。","澎湖、金門和馬祖。"),
    ("none","instruction_following/listing","列出兩種台灣的原住民族。","阿美族和泰雅族。"),
    ("none","instruction_following/listing","說出三種台式早午餐常見的飲料。","豆漿、米漿和紅茶。"),
    ("none","instruction_following/listing","列出三個台灣著名的古蹟或歷史建築。","赤崁樓、安平古堡和淡水紅毛城。"),
    ("none","instruction_following/listing","說出三種台灣的傳統布袋戲相關詞彙。","掌中戲、木偶和後場音樂。"),
    ("none","instruction_following/listing","列出兩個台灣的國家公園。","太魯閣國家公園和墾丁國家公園。"),
    ("none","instruction_following/listing","說出四種台灣常見的滷味食材。","豆干、海帶、蛋和米血。"),
    ("none","instruction_following/listing","列出三種台灣的傳統飲食文化。","辦桌、年夜飯和拜拜供品。"),

    # ══════════════════════════════════════════════
    # happy
    # ══════════════════════════════════════════════
    ("happy","acoustic_attributes/emotion/happy","用開心的語氣說：今天去夜市吃了超多好吃的！","今天去夜市吃了超多好吃的！"),
    ("happy","acoustic_attributes/emotion/happy","請高興地說：我買到了中華職棒總冠軍賽的門票！","我買到了中華職棒總冠軍賽的門票！"),
    ("happy","acoustic_attributes/emotion/happy","用興奮的語氣說：今天終於吃到了排隊兩小時的雞排！","今天終於吃到了排隊兩小時的雞排！"),
    ("happy","acoustic_attributes/emotion/happy","請愉快地說：颱風假放了三天，超開心的！","颱風假放了三天，超開心的！"),
    ("happy","acoustic_attributes/emotion/happy","用開心的語氣說：九份的夜景真的太美了！","九份的夜景真的太美了！"),
    ("happy","acoustic_attributes/emotion/happy","請高興地說：今天去淡水看夕陽，景色美極了！","今天去淡水看夕陽，景色美極了！"),
    ("happy","acoustic_attributes/emotion/happy","用興奮的語氣說：珍珠奶茶買一送一，今天真幸運！","珍珠奶茶買一送一，今天真幸運！"),
    ("happy","acoustic_attributes/emotion/happy","請愉快地說：廟會的陣頭表演精彩極了！","廟會的陣頭表演精彩極了！"),
    ("happy","acoustic_attributes/emotion/happy","用開心的語氣說：今天用YouBike騎了一圈大稻埕，好愜意！","今天用YouBike騎了一圈大稻埕，好愜意！"),
    ("happy","acoustic_attributes/emotion/happy","請高興地說：鳳梨酥試吃完馬上決定買十盒帶回家！","鳳梨酥試吃完馬上決定買十盒帶回家！"),
    ("happy","acoustic_attributes/emotion/happy","用興奮的語氣說：花蓮的海景民宿訂到了，期待暑假！","花蓮的海景民宿訂到了，期待暑假！"),
    ("happy","acoustic_attributes/emotion/happy","請愉快地說：烏來的溫泉泡完，整個人都放鬆了！","烏來的溫泉泡完，整個人都放鬆了！"),
    ("happy","acoustic_attributes/emotion/happy","用開心的語氣說：台東的星空好美，一輩子沒看過這麼多星星！","台東的星空好美，一輩子沒看過這麼多星星！"),
    ("happy","acoustic_attributes/emotion/happy","請高興地說：終於搶到了跨年演唱會的票！","終於搶到了跨年演唱會的票！"),
    ("happy","acoustic_attributes/emotion/happy","用興奮的語氣說：平溪天燈節的燈光飛上天的那一刻，感動到想哭！","平溪天燈節的燈光飛上天的那一刻，感動到想哭！"),
    ("happy","acoustic_attributes/emotion/happy","請愉快地說：今天搶到了限量的鳳梨酥禮盒！","今天搶到了限量的鳳梨酥禮盒！"),
    ("happy","acoustic_attributes/emotion/happy","用開心的語氣說：澎湖的海水真的藍到不像話！","澎湖的海水真的藍到不像話！"),
    ("happy","acoustic_attributes/emotion/happy","請高興地說：中華隊打進世界棒球經典賽決賽了！","中華隊打進世界棒球經典賽決賽了！"),
    ("happy","acoustic_attributes/emotion/happy","用興奮的語氣說：台南的擔仔麵真的是一絕！","台南的擔仔麵真的是一絕！"),
    ("happy","acoustic_attributes/emotion/happy","請愉快地說：今天在故宮看到了翠玉白菜，值回票價了！","今天在故宮看到了翠玉白菜，值回票價了！"),
    ("happy","acoustic_attributes/emotion/happy","用開心的語氣說：YouBike借還車好方便，通勤省了很多時間！","YouBike借還車好方便，通勤省了很多時間！"),
    ("happy","acoustic_attributes/emotion/happy","請高興地說：太魯閣的山景讓我整個心情都開朗了！","太魯閣的山景讓我整個心情都開朗了！"),
    ("happy","acoustic_attributes/emotion/happy","用興奮的語氣說：今年元宵節的燈籠比賽，我們社區拿了第一名！","今年元宵節的燈籠比賽，我們社區拿了第一名！"),
    ("happy","acoustic_attributes/emotion/happy","請愉快地說：金門的高粱酒買回來送禮，大家都好喜歡！","金門的高粱酒買回來送禮，大家都好喜歡！"),
    ("happy","acoustic_attributes/emotion/happy","用開心的語氣說：今天在西門町遇到偶像，還合照了！","今天在西門町遇到偶像，還合照了！"),

    # ══════════════════════════════════════════════
    # angry
    # ══════════════════════════════════════════════
    ("angry","acoustic_attributes/emotion/angry","用生氣的語氣說：捷運上大聲講電話真的很不禮貌！","捷運上大聲講電話真的很不禮貌！"),
    ("angry","acoustic_attributes/emotion/angry","用憤怒的語氣說：颱風假說好要放，結果最後取消了！","颱風假說好要放，結果最後取消了！"),
    ("angry","acoustic_attributes/emotion/angry","用生氣的語氣說：夜市那個攤販的態度真的太差了！","夜市那個攤販的態度真的太差了！"),
    ("angry","acoustic_attributes/emotion/angry","帶著憤怒的語氣說：國道塞車塞了三個小時！","國道塞車塞了三個小時！"),
    ("angry","acoustic_attributes/emotion/angry","用生氣的語氣說：排了一個小時的隊，結果賣完了！","排了一個小時的隊，結果賣完了！"),
    ("angry","acoustic_attributes/emotion/angry","用憤怒的語氣說：騎YouBike找不到停車位，氣死我了！","騎YouBike找不到停車位，氣死我了！"),
    ("angry","acoustic_attributes/emotion/angry","帶著憤怒說：廟會的鞭炮聲吵了一整晚，完全沒辦法睡覺！","廟會的鞭炮聲吵了一整晚，完全沒辦法睡覺！"),
    ("angry","acoustic_attributes/emotion/angry","用生氣的語氣說：高鐵誤點了四十分鐘，害我遲到！","高鐵誤點了四十分鐘，害我遲到！"),
    ("angry","acoustic_attributes/emotion/angry","用憤怒的語氣說：超商的店員態度很差，一點服務精神都沒有！","超商的店員態度很差，一點服務精神都沒有！"),
    ("angry","acoustic_attributes/emotion/angry","帶著憤怒的語氣說：那家餐廳的餐點等了一個小時還沒來！","那家餐廳的餐點等了一個小時還沒來！"),
    ("angry","acoustic_attributes/emotion/angry","用生氣的語氣說：颱風過後垃圾滿地，也沒人清理！","颱風過後垃圾滿地，也沒人清理！"),
    ("angry","acoustic_attributes/emotion/angry","用憤怒的語氣說：夜市停車場收費這麼貴，真的太誇張了！","夜市停車場收費這麼貴，真的太誇張了！"),
    ("angry","acoustic_attributes/emotion/angry","帶著憤怒說：春節期間高速公路塞到一動也不動！","春節期間高速公路塞到一動也不動！"),
    ("angry","acoustic_attributes/emotion/angry","用生氣的語氣說：那個觀光景點的票價一直漲，根本搶錢！","那個觀光景點的票價一直漲，根本搶錢！"),
    ("angry","acoustic_attributes/emotion/angry","用憤怒的語氣說：捷運月台上有人一直插隊，真的很火大！","捷運月台上有人一直插隊，真的很火大！"),
    ("angry","acoustic_attributes/emotion/angry","帶著憤怒說：這次颱風停班停課的標準搞不清楚，害我白跑一趟！","這次颱風停班停課的標準搞不清楚，害我白跑一趟！"),
    ("angry","acoustic_attributes/emotion/angry","用生氣的語氣說：手搖飲料的糖量標示根本不準確，害我踩雷！","手搖飲料的糖量標示根本不準確，害我踩雷！"),
    ("angry","acoustic_attributes/emotion/angry","用憤怒的語氣說：路上的機車騎得這麼猛，把我嚇了一大跳！","路上的機車騎得這麼猛，把我嚇了一大跳！"),
    ("angry","acoustic_attributes/emotion/angry","帶著憤怒的語氣說：夜市那家攤販份量縮水，價格還漲了！","夜市那家攤販份量縮水，價格還漲了！"),
    ("angry","acoustic_attributes/emotion/angry","用生氣的語氣說：廟會繞境把整條街堵死，害我遲到！","廟會繞境把整條街堵死，害我遲到！"),

    # ══════════════════════════════════════════════
    # sad
    # ══════════════════════════════════════════════
    ("sad","acoustic_attributes/emotion/sad","用悲傷的語氣說：中華隊在關鍵時刻輸了，真可惜。","中華隊在關鍵時刻輸了，真可惜。"),
    ("sad","acoustic_attributes/emotion/sad","悲傷地說：夜市那家開了二十年的老店終於關門了。","夜市那家開了二十年的老店終於關門了。"),
    ("sad","acoustic_attributes/emotion/sad","用難過的語氣說：颱風把阿里山的神木吹倒了。","颱風把阿里山的神木吹倒了。"),
    ("sad","acoustic_attributes/emotion/sad","悲傷地說：今年的天燈節因為下大雨，提前結束了。","今年的天燈節因為下大雨，提前結束了。"),
    ("sad","acoustic_attributes/emotion/sad","用難過的語氣說：九份那家百年豆腐店的老闆走了，回憶也跟著消逝了。","九份那家百年豆腐店的老闆走了，回憶也跟著消逝了。"),
    ("sad","acoustic_attributes/emotion/sad","悲傷地說：颱風過後，墾丁的珊瑚礁受損嚴重。","颱風過後，墾丁的珊瑚礁受損嚴重。"),
    ("sad","acoustic_attributes/emotion/sad","用難過的語氣說：台灣棒球的老教練退休了，一個時代結束了。","台灣棒球的老教練退休了，一個時代結束了。"),
    ("sad","acoustic_attributes/emotion/sad","悲傷地說：那條我從小走到大的老街，大半都被拆掉蓋大樓了。","那條我從小走到大的老街，大半都被拆掉蓋大樓了。"),
    ("sad","acoustic_attributes/emotion/sad","用難過的語氣說：鹿港的老師傅走了，傳統的糊紙技藝快要失傳了。","鹿港的老師傅走了，傳統的糊紙技藝快要失傳了。"),
    ("sad","acoustic_attributes/emotion/sad","悲傷地說：因為都市更新，兒時玩耍的那塊空地要蓋大樓了。","因為都市更新，兒時玩耍的那塊空地要蓋大樓了。"),
    ("sad","acoustic_attributes/emotion/sad","用難過的語氣說：因為旱災，日月潭的水位下降到幾十年來的最低點。","因為旱災，日月潭的水位下降到幾十年來的最低點。"),
    ("sad","acoustic_attributes/emotion/sad","悲傷地說：從小吃到大的那家傳統豆花店結束營業了。","從小吃到大的那家傳統豆花店結束營業了。"),
    ("sad","acoustic_attributes/emotion/sad","用難過的語氣說：台灣的棒球明星因傷退出國際賽，好可惜。","台灣的棒球明星因傷退出國際賽，好可惜。"),
    ("sad","acoustic_attributes/emotion/sad","悲傷地說：廟裡的那棵百年老樹，因為颱風倒塌了。","廟裡的那棵百年老樹，因為颱風倒塌了。"),
    ("sad","acoustic_attributes/emotion/sad","用難過的語氣說：澎湖的漁民說今年魚獲量比往年少了一半。","澎湖的漁民說今年魚獲量比往年少了一半。"),

    # ══════════════════════════════════════════════
    # surprised
    # ══════════════════════════════════════════════
    ("surprised","acoustic_attributes/emotion/surprised","用驚訝的語氣說：台北一○一的跨年煙火今年比往年還要精彩！","台北一○一的跨年煙火今年比往年還要精彩！"),
    ("surprised","acoustic_attributes/emotion/surprised","驚訝地說：這家隱藏版的台式早餐店排隊要等兩個小時！","這家隱藏版的台式早餐店排隊要等兩個小時！"),
    ("surprised","acoustic_attributes/emotion/surprised","用驚訝的語氣說：沒想到這個夜市攤販竟然獲得米其林推薦！","沒想到這個夜市攤販竟然獲得米其林推薦！"),
    ("surprised","acoustic_attributes/emotion/surprised","驚訝地說：日月潭的清晨霧景，真的美得不像話！","日月潭的清晨霧景，真的美得不像話！"),
    ("surprised","acoustic_attributes/emotion/surprised","用驚訝的語氣說：台灣的珍珠奶茶竟然在法國的超市都買得到！","台灣的珍珠奶茶竟然在法國的超市都買得到！"),
    ("surprised","acoustic_attributes/emotion/surprised","驚訝地說：平溪那條天燈老街居然還保留著百年前的樣貌！","平溪那條天燈老街居然還保留著百年前的樣貌！"),
    ("surprised","acoustic_attributes/emotion/surprised","用驚訝的語氣說：台積電的市值竟然已經超過好幾個國家的GDP！","台積電的市值竟然已經超過好幾個國家的GDP！"),
    ("surprised","acoustic_attributes/emotion/surprised","驚訝地說：這家路邊的碳烤玉米攤，在網路上有幾十萬個追蹤者！","這家路邊的碳烤玉米攤，在網路上有幾十萬個追蹤者！"),
    ("surprised","acoustic_attributes/emotion/surprised","用驚訝的語氣說：台灣便利商店的密度居然是全球第一！","台灣便利商店的密度居然是全球第一！"),
    ("surprised","acoustic_attributes/emotion/surprised","驚訝地說：原來台灣高山上的茶，可以在零下低溫中生長！","原來台灣高山上的茶，可以在零下低溫中生長！"),
    ("surprised","acoustic_attributes/emotion/surprised","用驚訝的語氣說：太魯閣的峽谷居然是幾百萬年前地殼隆起形成的！","太魯閣的峽谷居然是幾百萬年前地殼隆起形成的！"),
    ("surprised","acoustic_attributes/emotion/surprised","驚訝地說：澎湖的玄武岩柱狀地形，竟然和冰島的景觀這麼像！","澎湖的玄武岩柱狀地形，竟然和冰島的景觀這麼像！"),
    ("surprised","acoustic_attributes/emotion/surprised","用驚訝的語氣說：台灣阿里山的小火車居然有一百多年的歷史了！","台灣阿里山的小火車居然有一百多年的歷史了！"),
    ("surprised","acoustic_attributes/emotion/surprised","驚訝地說：台南安平的豆花老店，居然用的是百年前的古方！","台南安平的豆花老店，居然用的是百年前的古方！"),
    ("surprised","acoustic_attributes/emotion/surprised","用驚訝的語氣說：沒想到台東的熱氣球嘉年華一年比一年更有名！","沒想到台東的熱氣球嘉年華一年比一年更有名！"),

    # ══════════════════════════════════════════════
    # fearful
    # ══════════════════════════════════════════════
    ("fearful","acoustic_attributes/emotion/fearful","用害怕的語氣說：颱風來了，窗外的風聲讓我很不安。","颱風來了，窗外的風聲讓我很不安。"),
    ("fearful","acoustic_attributes/emotion/fearful","用恐懼的語氣說：聽說那條巷子在清明節前後特別不對勁。","聽說那條巷子在清明節前後特別不對勁。"),
    ("fearful","acoustic_attributes/emotion/fearful","用害怕的語氣說：地震警報突然響起，我嚇了一大跳。","地震警報突然響起，我嚇了一大跳。"),
    ("fearful","acoustic_attributes/emotion/fearful","用恐懼的語氣說：颱風眼通過的那一刻，整棟樓都在搖。","颱風眼通過的那一刻，整棟樓都在搖。"),
    ("fearful","acoustic_attributes/emotion/fearful","用害怕的語氣說：九份的山路在大霧裡完全看不到前面，好可怕。","九份的山路在大霧裡完全看不到前面，好可怕。"),
    ("fearful","acoustic_attributes/emotion/fearful","用恐懼的語氣說：那個廟會的鬼神面具讓我做了一夜惡夢。","那個廟會的鬼神面具讓我做了一夜惡夢。"),
    ("fearful","acoustic_attributes/emotion/fearful","用害怕的語氣說：半夜在便利商店遇到奇怪的陌生人，我趕快離開了。","半夜在便利商店遇到奇怪的陌生人，我趕快離開了。"),
    ("fearful","acoustic_attributes/emotion/fearful","用恐懼的語氣說：花蓮地震的規模這麼大，讓我一整夜沒睡著。","花蓮地震的規模這麼大，讓我一整夜沒睡著。"),
    ("fearful","acoustic_attributes/emotion/fearful","用害怕的語氣說：颱風警報一升高，我就緊張得睡不著。","颱風警報一升高，我就緊張得睡不著。"),
    ("fearful","acoustic_attributes/emotion/fearful","用恐懼的語氣說：夜市那條昏暗的小巷感覺跟鬼片場景一樣。","夜市那條昏暗的小巷感覺跟鬼片場景一樣。"),
    ("fearful","acoustic_attributes/emotion/fearful","用害怕的語氣說：颱風把路樹整棵連根拔起，實在太嚇人了。","颱風把路樹整棵連根拔起，實在太嚇人了。"),
    ("fearful","acoustic_attributes/emotion/fearful","用恐懼的語氣說：聽說七月是鬼月，晚上在外面真的很不安心。","聽說七月是鬼月，晚上在外面真的很不安心。"),

    # ══════════════════════════════════════════════
    # disgusted
    # ══════════════════════════════════════════════
    ("disgusted","acoustic_attributes/emotion/disgusted","用嫌惡的語氣說：夜市那攤臭豆腐的油感覺好久沒換了。","夜市那攤臭豆腐的油感覺好久沒換了。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","帶著厭惡的語氣說：捷運座位旁的垃圾就這樣丟在那裡，真的很不文明。","捷運座位旁的垃圾就這樣丟在那裡，真的很不文明。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","用嫌惡的語氣說：景區到處是遊客亂丟的垃圾，真讓人搖頭。","景區到處是遊客亂丟的垃圾，真讓人搖頭。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","帶著厭惡說：那個攤販明明份量縮水，還理直氣壯。","那個攤販明明份量縮水，還理直氣壯。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","用嫌惡的語氣說：便利商店的微波便當居然放了三天還在架上賣。","便利商店的微波便當居然放了三天還在架上賣。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","帶著厭惡的語氣說：夜市那家蚵仔煎用的蚵仔看起來不新鮮。","夜市那家蚵仔煎用的蚵仔看起來不新鮮。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","用嫌惡的語氣說：廟會結束後地上的香燭灰和垃圾沒人清，真的很髒。","廟會結束後地上的香燭灰和垃圾沒人清，真的很髒。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","帶著厭惡的語氣說：夜市的公廁環境真的讓人不敢進去。","夜市的公廁環境真的讓人不敢進去。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","用嫌惡的語氣說：路邊攤的滷味鍋看起來幾個月沒換過水。","路邊攤的滷味鍋看起來幾個月沒換過水。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","帶著厭惡的語氣說：觀光區那家店賣的伴手禮根本是工廠大量製造的，完全沒有誠意。","觀光區那家店賣的伴手禮根本是工廠大量製造的，完全沒有誠意。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","用嫌惡的語氣說：那攤炸物的油煙味嗆到我整個食欲都沒了。","那攤炸物的油煙味嗆到我整個食欲都沒了。"),
    ("disgusted","acoustic_attributes/emotion/disgusted","帶著厭惡說：捷運上有人脫鞋子，整個車廂都是腳臭味。","捷運上有人脫鞋子，整個車廂都是腳臭味。"),

    # ══════════════════════════════════════════════
    # whisper
    # ══════════════════════════════════════════════
    ("whisper","acoustic_attributes/volume/whisper","請悄悄地說：聽說那家新開的手搖飲超好喝，但要排很久。","聽說那家新開的手搖飲超好喝，但要排很久。"),
    ("whisper","acoustic_attributes/volume/whisper","悄悄告訴我：捷運上有個人一直在偷看別人的手機螢幕。","捷運上有個人一直在偷看別人的手機螢幕。"),
    ("whisper","acoustic_attributes/volume/whisper","請用悄悄話說：我找到一個超隱密的夜市美食攤，每次都要很早去才搶得到。","我找到一個超隱密的夜市美食攤，每次都要很早去才搶得到。"),
    ("whisper","acoustic_attributes/volume/whisper","悄悄地說：廟裡剛求到的籤說近期會有好事發生。","廟裡剛求到的籤說近期會有好事發生。"),
    ("whisper","acoustic_attributes/volume/whisper","請悄悄說：聽說台積電下個月要宣布重大消息。","聽說台積電下個月要宣布重大消息。"),
    ("whisper","acoustic_attributes/volume/whisper","悄悄告訴我：那個明星今天下午出現在西門町的咖啡廳。","那個明星今天下午出現在西門町的咖啡廳。"),
    ("whisper","acoustic_attributes/volume/whisper","請用悄悄話說：這家店的隱藏菜單要跟老闆說暗語才能點。","這家店的隱藏菜單要跟老闆說暗語才能點。"),
    ("whisper","acoustic_attributes/volume/whisper","悄悄說：媽祖託夢給老闆，說明年生意會特別好。","媽祖託夢給老闆，說明年生意會特別好。"),
    ("whisper","acoustic_attributes/volume/whisper","請悄悄地說：據說那個夜市攤販是藝人的親戚開的。","據說那個夜市攤販是藝人的親戚開的。"),
    ("whisper","acoustic_attributes/volume/whisper","悄悄告訴我：隔壁鄰居說他去年買了台積電的股票，賺了不少。","隔壁鄰居說他去年買了台積電的股票，賺了不少。"),
    ("whisper","acoustic_attributes/volume/whisper","請用悄悄話說：聽說這個廟的籤詩特別靈，很多人大老遠跑來求。","聽說這個廟的籤詩特別靈，很多人大老遠跑來求。"),
    ("whisper","acoustic_attributes/volume/whisper","悄悄說：有人說士林夜市深處有一攤沒有招牌的臭豆腐，在地人才知道。","有人說士林夜市深處有一攤沒有招牌的臭豆腐，在地人才知道。"),
    ("whisper","acoustic_attributes/volume/whisper","請悄悄地說：我朋友說他在九份的老街找到了一家非常便宜又好吃的芋圓店。","我朋友說他在九份的老街找到了一家非常便宜又好吃的芋圓店。"),
    ("whisper","acoustic_attributes/volume/whisper","悄悄告訴我：聽說那棟大樓的頂樓有一個對外開放的秘密花園。","聽說那棟大樓的頂樓有一個對外開放的秘密花園。"),
    ("whisper","acoustic_attributes/volume/whisper","請用悄悄話說：那個便利商店限量的聯名商品，凌晨一點補貨。","那個便利商店限量的聯名商品，凌晨一點補貨。"),

    # ══════════════════════════════════════════════
    # slow
    # ══════════════════════════════════════════════
    ("slow","acoustic_attributes/speed/slow","慢慢地說：九份的山路蜿蜒曲折，但沿途風景讓人心曠神怡。","九份的山路蜿蜒曲折，但沿途風景讓人心曠神怡。"),
    ("slow","acoustic_attributes/speed/slow","請放慢速度說：日月潭的湖面在清晨總是被薄霧輕輕覆蓋著。","日月潭的湖面在清晨總是被薄霧輕輕覆蓋著。"),
    ("slow","acoustic_attributes/speed/slow","用很慢的語調說：每年的媽祖遶境，信徒們虔誠跟隨、步步叩首。","每年的媽祖遶境，信徒們虔誠跟隨、步步叩首。"),
    ("slow","acoustic_attributes/speed/slow","慢慢地說：台南的古城巷弄裡，藏著許多百年歷史的老廟。","台南的古城巷弄裡，藏著許多百年歷史的老廟。"),
    ("slow","acoustic_attributes/speed/slow","請放慢速度說：花蓮的太魯閣，峽谷兩壁高聳，讓人感受到大自然的壯闊。","花蓮的太魯閣，峽谷兩壁高聳，讓人感受到大自然的壯闊。"),
    ("slow","acoustic_attributes/speed/slow","用很慢的語調說：台東的池上稻田，金黃色的稻浪隨風輕輕搖曳。","台東的池上稻田，金黃色的稻浪隨風輕輕搖曳。"),
    ("slow","acoustic_attributes/speed/slow","慢慢地說：平溪的天燈緩緩升起，帶著人們的心願飄向夜空。","平溪的天燈緩緩升起，帶著人們的心願飄向夜空。"),
    ("slow","acoustic_attributes/speed/slow","請放慢速度說：淡水老街在夕陽餘暉的映照下，顯得格外悠閒寧靜。","淡水老街在夕陽餘暉的映照下，顯得格外悠閒寧靜。"),
    ("slow","acoustic_attributes/speed/slow","用很慢的語調說：阿里山的小火車緩緩穿越茂密的森林，讓人彷彿回到了過去。","阿里山的小火車緩緩穿越茂密的森林，讓人彷彿回到了過去。"),
    ("slow","acoustic_attributes/speed/slow","慢慢地說：澎湖的海風輕輕吹過，吹散了夏日的暑氣。","澎湖的海風輕輕吹過，吹散了夏日的暑氣。"),
    ("slow","acoustic_attributes/speed/slow","請放慢速度說：廟裡的香火裊裊升起，信徒們默默低頭祈願。","廟裡的香火裊裊升起，信徒們默默低頭祈願。"),
    ("slow","acoustic_attributes/speed/slow","用很慢的語調說：鹿港老街的石板路，每一塊都承載著幾百年的歷史記憶。","鹿港老街的石板路，每一塊都承載著幾百年的歷史記憶。"),

    # ══════════════════════════════════════════════
    # fast
    # ══════════════════════════════════════════════
    ("fast","acoustic_attributes/speed/fast","請快速地說：台積電是台灣半導體產業最重要的企業之一。","台積電是台灣半導體產業最重要的企業之一。"),
    ("fast","acoustic_attributes/speed/fast","快速唸出：誠品書店、台北一○一、西門町是台北熱門觀光景點。","誠品書店、台北一○一、西門町是台北熱門觀光景點。"),
    ("fast","acoustic_attributes/speed/fast","請快速地唸出：珍珠奶茶、仙草凍、芋圓是台灣人氣甜品。","珍珠奶茶、仙草凍、芋圓是台灣人氣甜品。"),
    ("fast","acoustic_attributes/speed/fast","請快速說：捷運、公車、YouBike是台北最常用的交通方式。","捷運、公車、YouBike是台北最常用的交通方式。"),
    ("fast","acoustic_attributes/speed/fast","快速唸出：台北、新北、桃園、台中、台南、高雄。","台北、新北、桃園、台中、台南、高雄。"),
    ("fast","acoustic_attributes/speed/fast","請快速地說：鳳梨酥、牛軋糖、太陽餅是熱門的台灣伴手禮。","鳳梨酥、牛軋糖、太陽餅是熱門的台灣伴手禮。"),
    ("fast","acoustic_attributes/speed/fast","快速唸出：士林、饒河、寧夏、通化、南機場是台北的夜市。","士林、饒河、寧夏、通化、南機場是台北的夜市。"),
    ("fast","acoustic_attributes/speed/fast","請快速說：蚵仔煎、滷肉飯、鹽酥雞、臭豆腐、大腸包小腸。","蚵仔煎、滷肉飯、鹽酥雞、臭豆腐、大腸包小腸。"),
    ("fast","acoustic_attributes/speed/fast","快速唸出：阿里山、日月潭、太魯閣、墾丁、澎湖。","阿里山、日月潭、太魯閣、墾丁、澎湖。"),
    ("fast","acoustic_attributes/speed/fast","請快速地說：高鐵左營、台中、板橋、台北依序停靠。","高鐵左營、台中、板橋、台北依序停靠。"),
    ("fast","acoustic_attributes/speed/fast","快速唸出：元宵節、清明節、端午節、七夕、中秋節、冬至。","元宵節、清明節、端午節、七夕、中秋節、冬至。"),
    ("fast","acoustic_attributes/speed/fast","請快速說：釋迦、蓮霧、楊桃、芒果、鳳梨是台灣的特產水果。","釋迦、蓮霧、楊桃、芒果、鳳梨是台灣的特產水果。"),

    # ══════════════════════════════════════════════
    # multi_step
    # ══════════════════════════════════════════════
    ("none","instruction_following/multi_step","先說「我好餓」，然後說出你今晚想吃什麼台灣小吃。","我好餓。我想吃鹽酥雞。"),
    ("none","instruction_following/multi_step","先說「颱風要來了」，然後說出你要準備什麼。","颱風要來了。我要準備泡麵和手電筒。"),
    ("none","instruction_following/multi_step","先告訴我今天的天氣，再說你想去哪個夜市。","今天天氣晴朗。我想去饒河夜市。"),
    ("none","instruction_following/multi_step","先說出你最喜歡的台灣小吃，然後說明原因。","我最喜歡的是鹹酥雞。因為外酥內嫩，每次吃都很滿足。"),
    ("none","instruction_following/multi_step","先說今天星期幾，然後說出你週末打算去哪個景點。","今天是星期五。我打算週末去九份看夜景。"),

    # ══════════════════════════════════════════════
    # required_word
    # ══════════════════════════════════════════════
    ("none","instruction_following/required_word","請說一句包含「夜市」這個詞的句子。","今晚我要去士林夜市吃蚵仔煎。"),
    ("none","instruction_following/required_word","說一句包含「捷運」這個詞的句子。","台北的捷運班班準時，非常方便。"),
    ("none","instruction_following/required_word","請說一句包含「珍珠奶茶」這個詞的句子。","喝一杯珍珠奶茶是我每天下午的小確幸。"),
    ("none","instruction_following/required_word","說一句包含「颱風」的句子。","颱風來臨前，大家趕緊去超市補貨。"),
    ("none","instruction_following/required_word","請說一句包含「廟會」的句子。","廟會的陣頭鑼鼓聲響徹整條街。"),
    ("none","instruction_following/required_word","說一句包含「高鐵」的句子。","搭高鐵從台北到高雄只需要九十分鐘。"),
    ("none","instruction_following/required_word","請說一句包含「鳳梨酥」的句子。","鳳梨酥是台灣最受歡迎的伴手禮之一。"),
    ("none","instruction_following/required_word","說一句包含「媽祖」的句子。","每年的媽祖遶境都吸引數十萬信徒參與。"),

    # ══════════════════════════════════════════════
    # short_description
    # ══════════════════════════════════════════════
    ("none","instruction_following/short_description","描述一個台灣的夏天夜市。","熱鬧的攤販、油炸的香氣，人潮在燈光下川流不息。"),
    ("none","instruction_following/short_description","描述一個颱風天的台灣街頭。","狂風大雨，招牌搖晃，街上空無一人，只有便利商店燈火通明。"),
    ("none","instruction_following/short_description","描述台北捷運的早高峰。","車廂擁擠，每個人手握著吊環，安靜地滑著手機。"),
    ("none","instruction_following/short_description","描述一個廟會的場景。","震天的鑼鼓聲中，陣頭隊伍緩緩前行，香煙裊裊升起。"),
    ("none","instruction_following/short_description","描述阿里山清晨的雲海。","白色雲海在山谷間緩緩流動，日出的金光將雲層染成橘紅。"),
    ("none","instruction_following/short_description","描述台南夏日的午後。","烈日下老街的石板路上蒸騰著熱氣，古廟屋簷下有老人在乘涼。"),
    ("none","instruction_following/short_description","描述平溪放天燈的畫面。","一盞盞橘紅色的天燈緩緩升起，帶著心願飄向夜空，壯觀而感人。"),
    ("none","instruction_following/short_description","描述台灣的便利商店。","二十四小時不打烊，可以繳費、取件、買熱食，幾乎什麼都能做。"),

    # ══════════════════════════════════════════════
    # format_constraint
    # ══════════════════════════════════════════════
    ("none","instruction_following/format_constraint","請只用兩個字回答：你最喜歡的台灣小吃是什麼？","鹽酥雞。"),
    ("none","instruction_following/format_constraint","請只用三個字回答：台灣最高的建築是什麼？","台北一○一。"),
    ("none","instruction_following/format_constraint","請只用一個詞回答：台灣最有名的飲料是什麼？","珍珠奶茶。"),
    ("none","instruction_following/format_constraint","請只用四個字回答：台灣的夜晚哪裡最熱鬧？","士林夜市。"),

    # ══════════════════════════════════════════════
    # comparison
    # ══════════════════════════════════════════════
    ("none","instruction_following/comparison","台北一○一和台中國家歌劇院，哪一個比較高？","台北一○一比較高。"),
    ("none","instruction_following/comparison","高鐵和台鐵，哪一個速度比較快？","高鐵比較快。"),
    ("none","instruction_following/comparison","阿里山和玉山，哪一座比較高？","玉山比較高。"),
    ("none","instruction_following/comparison","士林夜市和饒河夜市，哪一個比較大？","士林夜市比較大。"),
    ("none","instruction_following/comparison","珍珠奶茶和仙草奶茶，哪一個比較常見？","珍珠奶茶比較常見。"),

    # ══════════════════════════════════════════════
    # negative_constraint
    # ══════════════════════════════════════════════
    ("none","instruction_following/negative_constraint","說出三個台灣城市，但不要提到台北。","台中、高雄和台南。"),
    ("none","instruction_following/negative_constraint","列出三種台灣小吃，但不要說滷肉飯。","蚵仔煎、鹽酥雞和臭豆腐。"),
    ("none","instruction_following/negative_constraint","說出兩種台灣特產，不要提到鳳梨酥。","牛軋糖和太陽餅。"),
    ("none","instruction_following/negative_constraint","說出三個台灣景點，不要提到阿里山。","日月潭、太魯閣和墾丁。"),

    # ══════════════════════════════════════════════
    # word_extraction
    # ══════════════════════════════════════════════
    ("none","instruction_following/word_extraction","請只說出這句話的第一個字：珍珠奶茶真的很好喝。","珍。"),
    ("none","instruction_following/word_extraction","請只說出這句話的最後一個字：今天去夜市吃了很多東西。","西。"),
    ("none","instruction_following/word_extraction","請只說出這句話的第三個字：台灣的夜市很好逛。","的。"),

    # ══════════════════════════════════════════════
    # conditional
    # ══════════════════════════════════════════════
    ("none","instruction_following/conditional","如果是颱風天，請說「待在家」。現在是颱風天。","待在家。"),
    ("none","instruction_following/conditional","如果飲料是珍珠奶茶，請說「加珍珠」。這杯是珍珠奶茶。","加珍珠。"),
    ("none","instruction_following/conditional","如果是假日，請說「去夜市」。今天是假日。","去夜市。"),

    # ══════════════════════════════════════════════
    # replacement
    # ══════════════════════════════════════════════
    ("none","instruction_following/replacement","請把「夜市」換成「廟會」：我最喜歡去夜市逛街。","我最喜歡去廟會逛街。"),
    ("none","instruction_following/replacement","請把「捷運」換成「高鐵」：我每天搭捷運上班。","我每天搭高鐵上班。"),
    ("none","instruction_following/replacement","請把「珍珠奶茶」換成「仙草奶茶」：我今天喝了一杯珍珠奶茶。","我今天喝了一杯仙草奶茶。"),
    ("none","instruction_following/replacement","請把「阿里山」換成「日月潭」：我下週要去阿里山旅遊。","我下週要去日月潭旅遊。"),

    # ══════════════════════════════════════════════
    # completion
    # ══════════════════════════════════════════════
    ("none","instruction_following/completion","請完成這句話：台灣最有名的夜市是……","士林夜市。"),
    ("none","instruction_following/completion","請完成這句話：台灣人最喜歡的飲料是……","珍珠奶茶。"),
    ("none","instruction_following/completion","請完成這句話：台灣最高的山是……","玉山。"),

    # ══════════════════════════════════════════════
    # simple_arithmetic (台灣生活情境)
    # ══════════════════════════════════════════════
    ("none","instruction_following/simple_arithmetic","一杯珍珠奶茶五十五元，兩杯共多少元？","一百一十元。"),
    ("none","instruction_following/simple_arithmetic","夜市買了三樣小吃，各七十元，總共多少元？","兩百一十元。"),
    ("none","instruction_following/simple_arithmetic","高鐵票原價一千元，打八折是多少元？","八百元。"),
]


def main():
    if not Path(BACKUP).exists():
        shutil.copy2(INPUT, BACKUP)
        print(f"Backup created: {BACKUP}")
    else:
        shutil.copy2(BACKUP, INPUT)
        print(f"Restored from backup: {BACKUP}")

    data = [json.loads(l) for l in open(INPUT, encoding="utf-8")]
    print(f"Loaded {len(data)} examples")

    idx_map = defaultdict(list)
    for i, d in enumerate(data):
        idx_map[(d["style"], d["ability"])].append(i)

    rng = random.Random(42)
    replaced = 0
    skipped = 0
    for style, ability, instruction, target_text in TW_POOL:
        candidates = idx_map.get((style, ability), [])
        if not candidates:
            print(f"  [skip] no slot for style={style} ability={ability}")
            skipped += 1
            continue
        chosen = rng.choice(candidates)
        data[chosen]["instruction"] = instruction
        data[chosen]["target_text"] = target_text
        candidates.remove(chosen)
        replaced += 1

    with open(INPUT, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nDone: replaced {replaced} / {len(TW_POOL)} examples  (skipped {skipped})")


if __name__ == "__main__":
    main()
