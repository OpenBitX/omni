const zh = {
  betaTag: "内测版 · 三种视角",
  heroLine1: "让万物",
  heroSoul: "有灵魂",
  heroSubPrefix: "万物皆有声音 · ",
  heroSubMid: "选它教你的方式。",
  heroSubSuffix:
    " 把镜头对准这个世界；让它陪你玩、教你东西，或者告诉你谁曾在这里生活过。",
  ctaBegin: "开始 →",
  ctaSkip: "跳过，直接玩 →",
  demoKicker: "演示",
  demoTitle: "看它活过来",
  demoHint: "点 · 锁 · 听",
  clipsKicker: "实拍",
  clipsTitle: "大家都在玩",
  clipsHint: "真实用户 · 无剪辑",
  diagramKicker: "原理",
  diagramTitle: "一次点击如何变成声音",
  diagramHint: "设备端视觉 + Whisper · 云端角色 · 流式语音",
  openKicker: "开源",
  openTitle: "我们是开源的 ✿",
  openCopy:
    "每个模型、提示词和完整许可证都公开开放。尽情 fork、remix，给你身边的东西也装上灵魂。",
  openBtn: "在 GitHub 上找到",
  openFoot: "完整许可证 · 全部公开 · 都是你的",
  footerTagline: "万物拟人局 · everything has a voice",
  footerGallery: "画廊",
  footerBegin: "开始",
  visionKicker: "完整愿景",
  visionWarn: "黑客松 · 这部分未上线",
  visionTitle: "让万物有灵魂 ✿",
  visionLede:
    "你现在玩到的是娱乐视角。真正的产品是一个基础模型。",
  visionBody:
    "想法很简单：你身边的每件东西都有值得学习的地方。物理世界不是一堆沉默的死物——它是知识、语境和情感的宝库。我们想打造一个基础模型，把它变成随时可查询的知识源。你选择视角。切换到语言——你可以用正在学的语言和周围的东西对话：指着杯子让它教你怎么念，让它用西班牙语自我介绍，来来回回直到单词记住为止。你的房间变成教室。切换到历史——街角的长凳告诉你这条路一百年来的变迁。切换到玩耍——就是你现在用的。用户决定要提取什么。",
  visionClosing:
    "这个周末我们只来得及把基础架构做扎实——稳定、可靠、能用。剩下的一切再多一天就够了。",
  visionNextBadge: "下一步",
  visionNext:
    "基础模型已经开源——任何人都可以在上面构建下一个前沿。我们自己的下一步可能是一个教育应用：硬基础设施已经完成，剩下的就是针对特定场景做微调。",
} as const;

export default zh;
export type ZhKeys = keyof typeof zh;
