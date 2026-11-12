#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/ipv4-routing-helper.h"
#include "ns3/traffic-control-module.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/flow-monitor-module.h" 
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("MilanMooSimulation");

struct HourlyCommand {
    std::string timestamp;
    double gridCarbon;
    int nodeStates[30];
    double nodeTraffic[30];
    double nodeNetWatts[30]; 
};

std::vector<HourlyCommand> cityTraces;
NodeContainer milanNodes;
double totalBaselineKWh = 0.0;
double totalMooKWh = 0.0;
AnimationInterface* anim = nullptr; 

std::vector<Ptr<OnOffApplication>> vipApps;
std::vector<Ptr<OnOffApplication>> beApps;

std::vector<HourlyCommand> LoadTraceFile(std::string filename) {
    std::vector<HourlyCommand> traceData;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) NS_FATAL_ERROR("Could not open trace file: " << filename);

    std::getline(file, line); 
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        HourlyCommand cmd;

        std::getline(ss, cmd.timestamp, ',');
        std::getline(ss, cell, ','); cmd.gridCarbon = std::stod(cell);

        for (int i = 0; i < 30; ++i) {
            std::getline(ss, cell, ','); cmd.nodeStates[i] = std::stoi(cell);
            std::getline(ss, cell, ','); cmd.nodeTraffic[i] = std::stod(cell);
            std::getline(ss, cell, ','); cmd.nodeNetWatts[i] = std::stod(cell);
            std::getline(ss, cell, ','); 
            std::getline(ss, cell, ','); 
        }
        traceData.push_back(cmd);
    }
    file.close();
    return traceData;
}

void SetVipQoS(Ptr<Application> app) {
    Ptr<OnOffApplication> onoff = DynamicCast<OnOffApplication>(app);
    if (onoff && onoff->GetSocket()) {
        onoff->GetSocket()->SetIpTos(0xb8); 
    }
}

void UpdateCityStates(size_t hourIndex) {
    if (hourIndex >= cityTraces.size()) return;

    HourlyCommand currentCmd = cityTraces[hourIndex];
    std::cout << "\n[Time: " << currentCmd.timestamp << "] Grid Carbon: " << currentCmd.gridCarbon << " gCO2/kWh" << std::endl;

    for (uint32_t i = 0; i < milanNodes.GetN(); ++i) {
        Ptr<Node> node = milanNodes.Get(i);
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        int state = currentCmd.nodeStates[i];
        double traffic = currentCmd.nodeTraffic[i];

        // --- FIXED: Raised the minimum speed to 500 Kbps to prevent NS-3 Time-Warp Overflows at 4:00 AM ---
        // Lighter footprint: Core routers can handle the aggregation now
        double vipKbps = std::max(100.0, traffic * 0.5); 
        double beKbps = std::max(500.0, traffic * 3.0);

        if (i < vipApps.size() && i < beApps.size()) {
            vipApps[i]->SetAttribute("DataRate", DataRateValue(DataRate(vipKbps * 1000)));
            beApps[i]->SetAttribute("DataRate", DataRateValue(DataRate(beKbps * 1000)));
        }

        double baselineWatts = 100.0 + (traffic * 0.1); 
        totalBaselineKWh += (baselineWatts / 1000.0);
        totalMooKWh += (currentCmd.nodeNetWatts[i] / 1000.0);

        for (uint32_t iface = 1; iface < ipv4->GetNInterfaces(); ++iface) {
            Ptr<NetDevice> device = ipv4->GetNetDevice(iface);
            Ptr<PointToPointNetDevice> ptpDevice = DynamicCast<PointToPointNetDevice>(device);

            if (state == 0) {
                ipv4->SetUp(iface);
                if (ptpDevice) ptpDevice->SetDataRate(DataRate("100Mbps")); 
                if (anim) {
                    anim->UpdateNodeColor(node, 0, 255, 0); 
                    anim->UpdateNodeDescription(node, "Active (" + std::to_string((int)traffic) + ")");
                }
            } 
            else if (state == 1) {
                ipv4->SetUp(iface);
                if (ptpDevice) ptpDevice->SetDataRate(DataRate("25Mbps")); 
                if (anim) {
                    anim->UpdateNodeColor(node, 255, 165, 0); 
                    anim->UpdateNodeDescription(node, "Buffer (" + std::to_string((int)traffic) + ")");
                }
            } 
            else if (state == 2) {
                ipv4->SetDown(iface); 
                if (anim) {
                    anim->UpdateNodeColor(node, 255, 0, 0); 
                    anim->UpdateNodeDescription(node, "Sleep (" + std::to_string((int)traffic) + ")");
                }
            }
        }
    }
    Simulator::Schedule(Seconds(1.0), &UpdateCityStates, hourIndex + 1);
}

int main (int argc, char *argv[]) {
    int duration = 24; 
    CommandLine cmd;
    cmd.AddValue("duration", "Number of hours to simulate", duration);
    cmd.Parse (argc, argv);

    cityTraces = LoadTraceFile("scratch/ns3_moo_traces.csv");
    std::cout << "--- Milan MOO Digital Twin Simulation Starting ---" << std::endl;

    milanNodes.Create(30);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
    for (int i=0; i<8; i++) positionAlloc->Add(Vector(10 + (i%4)*15, 80 + (i/4)*15, 0.0));
    for (int i=8; i<16; i++) positionAlloc->Add(Vector(70 + (i%4)*15, 80 + (i/4)*15, 0.0));
    for (int i=16; i<24; i++) positionAlloc->Add(Vector(10 + (i%4)*15, 20 + (i/4)*15, 0.0));
    for (int i=24; i<30; i++) positionAlloc->Add(Vector(70 + (i%3)*15, 20 + (i/3)*15, 0.0));

    mobility.SetPositionAllocator (positionAlloc);
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (milanNodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
    p2p.SetChannelAttribute ("Delay", StringValue ("2ms"));

    OlsrHelper olsr; 
    InternetStackHelper stack;
    stack.SetRoutingHelper(olsr);
    stack.Install (milanNodes);

    Ipv4AddressHelper address;
    address.SetBase ("10.1.1.0", "255.255.255.252");

    TrafficControlHelper tch;
    tch.SetRootQueueDisc ("ns3::PfifoFastQueueDisc", "MaxSize", StringValue("10000p")); // 10x larger buffer!

    for (uint32_t i = 0; i < 30; ++i) {
        uint32_t nextNode = (i == 7 || i == 15 || i == 23 || i == 29) ? (i - (i%8)) : (i + 1);
        if (i == 29) nextNode = 24; 
        NetDeviceContainer link = p2p.Install (milanNodes.Get(i), milanNodes.Get(nextNode));
        tch.Install(link); 
        address.Assign (link);
        address.NewNetwork ();
    }
    uint32_t bridges[4][2] = {{3, 8}, {14, 29}, {24, 19}, {17, 0}};
    for (int b = 0; b < 4; b++) {
        NetDeviceContainer link = p2p.Install (milanNodes.Get(bridges[b][0]), milanNodes.Get(bridges[b][1]));
        tch.Install(link); address.Assign(link); address.NewNetwork();
    }

    uint16_t port = 9;
    for (uint32_t i = 0; i < 30; ++i) {
        PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sink.Install(milanNodes.Get(i));
        sinkApp.Start(Seconds(0.0)); sinkApp.Stop(Seconds(duration + 1));

        PacketSinkHelper sink2("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port+1));
        ApplicationContainer sinkApp2 = sink2.Install(milanNodes.Get(i));
        sinkApp2.Start(Seconds(0.0)); sinkApp2.Stop(Seconds(duration + 1));
    }

    for (uint32_t i = 0; i < 30; ++i) {
        uint32_t dst = (i + 15) % 30; 
        Ptr<Ipv4> ipv4Dst = milanNodes.Get(dst)->GetObject<Ipv4>();
        Ipv4Address dstIp = ipv4Dst->GetAddress(1, 0).GetLocal();

        OnOffHelper vipApp ("ns3::UdpSocketFactory", InetSocketAddress (dstIp, port));
        vipApp.SetAttribute ("PacketSize", UintegerValue (512));
        vipApp.SetAttribute ("DataRate", DataRateValue(DataRate("1Kbps"))); 
        
        OnOffHelper beApp ("ns3::UdpSocketFactory", InetSocketAddress (dstIp, port+1));
        beApp.SetAttribute ("PacketSize", UintegerValue (1024));
        beApp.SetAttribute ("DataRate", DataRateValue(DataRate("1Kbps"))); 
        
        ApplicationContainer a1 = vipApp.Install (milanNodes.Get (i));
        ApplicationContainer a2 = beApp.Install (milanNodes.Get (i));
        
        a1.Start (Seconds (1.0)); a1.Stop (Seconds (duration));
        a2.Start (Seconds (1.0)); a2.Stop (Seconds (duration));

        vipApps.push_back(DynamicCast<OnOffApplication>(a1.Get(0)));
        beApps.push_back(DynamicCast<OnOffApplication>(a2.Get(0)));

        Simulator::Schedule(Seconds(1.001), &SetVipQoS, a1.Get(0));
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    anim = new AnimationInterface("milan_moo_animation.xml");
    anim->EnablePacketMetadata(true);

    Simulator::Schedule(Seconds(0.0), &UpdateCityStates, 0);

    std::cout << "Running simulation for " << duration << " virtual hours..." << std::endl;
    Simulator::Stop (Seconds (duration + 1));
    Simulator::Run ();

    monitor->CheckForLostPackets ();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();

    uint64_t totalVipTx = 0, totalVipDropped = 0;
    uint64_t totalBeTx = 0, totalBeDropped = 0;

    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
        if (t.destinationPort == port) { 
            totalVipTx += i->second.txPackets;
            totalVipDropped += i->second.lostPackets;
        } else { 
            totalBeTx += i->second.txPackets;
            totalBeDropped += i->second.lostPackets;
        }
    }

    std::cout << "\n======================================================\n";
    std::cout << "         FINAL THESIS RESULTS: MULTI-OBJECTIVE        \n";
    std::cout << "======================================================\n";
    std::cout << "--- 1. ENERGY EFFICIENCY ---\n";
    std::cout << "Baseline Network Energy: " << totalBaselineKWh << " kWh\n";
    std::cout << "AI MOO Network Energy:   " << totalMooKWh << " kWh\n";
    std::cout << "Energy Savings:          " << ((totalBaselineKWh - totalMooKWh) / totalBaselineKWh) * 100.0 << " %\n\n";
    
    std::cout << "--- 2. QUALITY OF SERVICE (PACKET DROPS) ---\n";
    std::cout << "VIP Packets Sent:        " << totalVipTx << "\n";
    std::cout << "VIP Packets Dropped:     " << totalVipDropped << " (" << (totalVipTx > 0 ? (totalVipDropped * 100.0 / totalVipTx) : 0) << "%)\n";
    std::cout << "Best Effort Sent:        " << totalBeTx << "\n";
    std::cout << "Best Effort Dropped:     " << totalBeDropped << " (" << (totalBeTx > 0 ? (totalBeDropped * 100.0 / totalBeTx) : 0) << "%)\n";
    std::cout << "======================================================\n";

    delete anim;
    Simulator::Destroy ();
    return 0;
}