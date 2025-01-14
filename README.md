# atomistic-simulation-tutorial
このリポジトリは、PFCCによる[Atomistic simulation hands on tutorial on Matlantis](https://matlantis.com/)を、誰でも無料で使えるオープンソースソフトウェアだけで動かせるようにしたフォークです。  
これは個人が自分の勉強を兼ねた趣味として取り組んでいるプロジェクトであり、更新は少しずつになるかもしれません。内容についてのフィードバックや拡充のご提案・pull requestも歓迎しています。

## チュートリアルへのリンク
（準備中です。近日中に順次公開予定です）

## このチュートリアルの目的
このチュートリアルの目的は、誰もがNeural Network Potential（NNP）を用いたDFT計算や分子動力学（MD）計算について実際にコードを動かしながら学べるリソースとなることです。

上記のMatlantisチュートリアルでは、NNPを用いたDFT計算やMD計算について、実践的コードを示しながら非常にわかりやすく説明されており、AI時代の計算科学を学ぶために素晴らしい教材です。
しかし、法人向け商用製品であるMatlantisのチュートリアルという位置づけのため、オリジナルのコードを動かしながら学ぶためにはMatrantisの契約が必要で、学生や研究者が気軽に試すというわけにはいきませんでした。

そこで、オープンソースソフトウェア（OSS）として公開されているNNPを用いて、Matlantisなしで動かせるようにコードと説明を修正しました。
また、（Matlantisに限らない）一般のNNPを使った材料シミュレーションを学ぶリソースとなるよう、内容は適宜拡充していくつもりです。

## お気持ち
機械学習を用いて物理シミュレーションを加速するという試みは、今日の材料科学において最も活発に研究されている分野の一つです。NNPをはじめとするこのようなAI-boostedシミュレーションは、今後の材料科学に大きな変革を起こすツールとなるでしょう。そしてこのための道具はMatlantisだけではなく、OSSとして公開されているものも数多くあります。OSSを用いて多くの研究者がAI-boostedシミュレーションの力を手にすれば、材料研究は大きく加速すると確信しています。

商用製品のチュートリアルを誰でも動かせるようにすることは、商売の邪魔をしている印象を与えるかもしれませんが（実際、悩みました）、私はより多くの研究者がAI-boostedシミュレーションを日常的なツールとして使いこなせるようになるべきであり、そうなればMatlantisを必要とするユーザーも自然と増えると考え、公開することにしました。
現状で私が最も心配していることは、必要な情報の不足からNNPが普及せず、ごく一部の研究者の道具となってしまうことです。
資源枯渇や気候変動への対策が喫緊の課題となっている今日、材料科学の果たすべき役割は大きく、世界中の研究者が力を合わせて課題解決に取り組む必要があります。先にも書いたように、AI-boostedシミュレーションはこの突破口の一つとなり得るものであり、誰もが使えるようになるべき道具です。

私は、Matlantisを開発しているPFCCもこの志を共有していると考えます。その証拠に（？）オリジナルのMatlantisチュートリアルは、コード含め扱いやすいライセンス（CC-BY 4.0）で公開されています。このような貴重な資料を公開してくれたMatlantisの開発者の皆さんに敬意を払い、公平な記述となるよう注意したほか、Matlantis固有の機能や記述であっても、学習のために役立つと思われる部分はなるべく残すことにしました。

## 詳しい人向けの説明
Matlantis（のコア機能であるPFP）は物質シミュレーション用PythonパッケージであるASEのcalculatorモジュールとして提供されていることから、同様のAPIを備えたOSSなNNPを使うようにコードと説明を微修正しました。


## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
