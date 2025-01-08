import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def main():
    df = pd.read_csv("Cost_of_Living_Index_by_Country_2024.csv")
    """
    問(1) まず、このデータは欠損値がないかどうか調べることにします。欠損値のある列はありますか。あれば、その列を全て列記してください。
    => 欠損地無し
    """
    print("(1) 欠損地の確認")
    print(df.describe())
    print(df.info(), end="\n\n")

    """
    問(2) それぞれの指数がどれくらい関係しているか相関係数を調べようと思います。次のような相関係数をヒートマップにしてください。
    """
    # 相関係数行の作成
    corr = df.drop(["Country", "Rank"], axis=1).corr()
    fig1 = plt.figure()
    fig1.tight_layout()
    ax1 = fig1.add_subplot()
    sns.heatmap(corr, annot=True, cbar=True, cmap="coolwarm", ax=ax1)
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    # ax1.set_yticklabels(ax1.get_xticklabels(), rotation=45)
    # fig1.subplots_adjust(left=0.1, bottom=0.5)
    # ax1.set_xticklabels(ax1.get_xticklabels(), fonsize=8)
    # ax1.set_yticklabels(ax1.get_xticklabels(), fonsize=8)
    ax1.set_title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")

    """
    問(3) Cost of Living Index , Rent Index	Cost of Living Plus , Rent Index , 	Groceries Index , Restaurant Price Index , Local Purchasing Power Index の平均値を求めてください。
    """
    print("(3) Cost of Living Index , Rent Index, Cost of Living Plus , Rent Index , Groceries Index , Restaurant Price Index , Local Purchasing Power Index の平均値")
    for i in df.drop("Country", axis=1).columns:
        print(f"{i} => {df[i].mean()}")
    print()

    """
    問(4) Rent Indexの上位 10 か国を 1位XXX,  2位XXX,  …  のように列記してください。
    """
    rent_index_top_10 = df.sort_values("Rent Index", ascending=False).head(10)[["Country", "Rent Index"]].reset_index()
    print("(4) Rent Index上位10か国")
    [print(f"第{i+1}位 => {x["Country"]}: {x["Rent Index"]}") for i, x in rent_index_top_10.iterrows()]

    """
    問(5) 日本は、Cost of Living Index（生活費指数）で第何位ですか？
    """
    cli_sort = df.sort_values("Cost of Living Index")[["Country", "Cost of Living Index"]].reset_index(drop=True)
    print()
    print(f"(5) 日本は、Cost of Living Indexで第何位か？")
    search_country = "Japan"
    sc_cli_order = cli_sort.query("Country == @search_country").index[0]
    print(f"{search_country}は第{sc_cli_order}位", end="\n\n")

    """
    問(6) 家賃指数は、ある都市のアパートの賃貸価格をニューヨーク市と比較したものです。家賃指数が 80 の場合、その都市の平均賃貸価格はニューヨーク市よりも約 20% 低いことを示しています。
          日本の家賃は、この指数を使って、ニューヨーク市と比べて何％位　高い／低いでしょうか？「XXX％高い」のように答えてください。
    """
    search_country = "Japan"
    print(f"(6) {search_country}のニューヨーク市と比較した賃貸価格")
    sc_ri_order = df.query("Country == @search_country").iloc[0, 3]
    if sc_ri_order < 100:
        print(f"{search_country}はニューヨーク市に比べて賃貸価格が{100 - sc_ri_order}％低い")
    else:
        print(f"{search_country}はニューヨーク市に比べて賃貸価格が{sc_ri_order - 100}％高い")

    """
    問（7）
    Cost of Living Index,  Rent Index	Cost of Living Plus,  Rent Index, 	Groceries Index,  Restaurant Price Index,  Local Purchasing Power の６つのデータで各国を 5 つのグループに分けるため、クラスタリングをしてみます。
    k-means を使い、国を 5 つのグループに分けてみましょう。さらに、横軸をCost of Living Index,  縦軸に Local Purchasing Power  をとり、次のような散布図にしてください。
    """

    X = df.drop(["Country", "Rank"], axis=1).to_numpy()
    color_palettes = np.array(["red", "blue", "green", "purple", "orange"])
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(X)

    # 
    res_df = pd.DataFrame(df[["Cost of Living Index", "Local Purchasing Power Index"]])
    res_df["pred_label"] = model.labels_

    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot()

    for l in np.arange(5):
        plot_df = res_df.query("pred_label == @l")
        ax2.scatter(plot_df["Cost of Living Index"], plot_df["Local Purchasing Power Index"], c=color_palettes[l], label=l)
    
    ax2.set_title("Clustering of Country by Cost of Living Index and Local Purchasing Power Index")
    ax2.set_xlabel("Cost of Living Index")
    ax2.set_ylabel("Local Purchasing Power Index")
    ax2.legend()
    plt.savefig("scatter.png")

    """
    問（8）
    問（7）でクラスタリングした 5 つのグループをそれぞれどんなグループか、簡単に説明してください。
    例：生活費指数が高く、地元購買力が低いグループ
    """

    """
    Cost of Living Index(生活費指数)とは、ある国と比較したときの生活費の物価の高さ
    Local Purchasing Power(地元購買力)とは、特定の地域や国における人々の購買力、つまりその地域で得た収入でどれだけの財やサービスを購入できるかを示す指標

    ラベル0のクラス： 生活指数が半分ほどで地元購買力は同じかそれ以上の国が多いので最も自由に使えるお金が多そう。
    ラベル1のクラス： 生活指数、地元購買力ともに最も低いため、生活費は安いがその分給料も低くあまりお金に余裕がない国のグループと言えそう
    ラベル2のクラス： 生活指数が低く、地元購買力は同じくらいかそれ以上なので、ニューヨークよりも自由に使えるお金が多そう
    ラベル3のクラス： 生活指数は半分ほどで、地元購買力が30 ~ 80くらいなのでラベル4のクラスターよりも少しお金に余裕がないグループと言えそう。
    ラベル4のクラス： 生活指数は半分くらいで地元購買力は60 ~ 100くらいなので生活費は安いが給料もそれほど高くなくニューヨークほどお金に余裕がないグループと言えそう。
    """

if __name__ == "__main__":
    main()